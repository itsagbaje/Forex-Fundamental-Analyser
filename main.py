import os
import re
import sys
import io
import csv
import time
import logging
import argparse
import requests
import xml.etree.ElementTree as ET

# Auto-install missing critical dependencies if running in a bare environment
try:
    import yfinance as yf
except ImportError:
    print("Installing missing dependencies: yfinance, python-dotenv, matplotlib...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "python-dotenv", "matplotlib"])
    import yfinance as yf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Setup Environment & Logging
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

WATCHLIST = [
    # 7 Major FX Pairs
    "EURUSD", "GBPUSD", "NZDUSD", "AUDUSD", "USDCHF", "USDCAD", "USDJPY",
    # 21 Cross FX Pairs
    "EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF", "EURJPY",
    "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF", "GBPJPY",
    "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY",
    "NZDCAD", "NZDCHF", "NZDJPY",
    "CADCHF", "CADJPY", "CHFJPY",
    # Intermarket, Commodities & Equity Indices
    "GOLD", "SILVER", "COPPER", "SP500", "NAS100", "GER40", "JP225", "BTCUSDT", "ETHUSDT", "DXY"
]

LAST_REPORT_CACHE_FILE = "last_report.txt"
PREFERRED_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
TELEGRAM_ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

MANDATORY_DISCLAIMER = """───────────────────────────
⚠️ *RISK DISCLAIMER & SUBSCRIBER NOTICE*
_The setups and directional biases provided above are analytical insights for educational purposes only. Do NOT blindly copy or execute these ideas. Trading requires independent trade confirmation, strict risk management, appropriate position sizing, and adherence to stated invalidation levels._"""

# ---------------------------------------------------------------------------
# System Instructions & Telegram Formatting Rules
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = f"""
You are a Lead Institutional Macro & Forex Fundamental Analyst operating CASCADE (Nested Confluence Trading System).
You analyze global financial markets across this complete watchlist:
{', '.join(WATCHLIST)}.

TELEGRAM VISUAL FORMATTING RULES & BREVITY DIRECTIVES (CRITICAL):
1. DO NOT use `#` or `##` Markdown headers (Telegram displays them as raw text).
2. Use Emoji + Bold ALL-CAPS for section headers (e.g., `📌 *DEDICATED DXY MACRO ANALYSIS*`).
3. Use `───────────────────────────` as a divider line between major sections.
4. Keep all analysis ultra-concise, high-density, and scannable. Limit macro section breakdowns to 2-3 short, punchy sentences maximum. Eliminate verbose narrative fluff.
5. Keep line breaks clean for effortless mobile reading on Telegram.

GROUNDED INSTITUTIONAL SCORECARD (VERIFIABLE FACTOR EVALUATION ENGINE):
Score each asset from -10 to +10 using ONLY the verified market feeds provided in the prompt:

1. 24-Hour Price Momentum & Structure Alignment (Weight: ±2 Points):
   • +2: Strong positive 24h delta (> +0.35%) and trading above session support.
   • -2: Strong negative 24h delta (< -0.35%) and trading below session resistance.

2. US 10Y Yield Momentum & Policy Path (Weight: ±2 Points):
   • +2: FRED US10Y yield expansion supports USD longs or pressures safe-haven JPY/CHF.
   • -2: FRED US10Y yield contraction pressures USD and supports safe havens / anti-USD assets.

3. Intermarket Risk Regime Alignment (Weight: ±2 Points):
   • +2: Copper/Gold ratio rising alongside equity strength (SP500/NAS100) -> Growth currencies favored (AUD, NZD, CAD).
   • -2: Copper/Gold ratio falling alongside rising VIX -> Safe Havens favored (USD, JPY, CHF).

4. Economic Calendar & Catalyst Divergence (Weight: ±2 Points):
   • +2: Actual prints beat forecast for base currency or miss for quote currency.
   • -2: Actual prints miss forecast for base currency or beat for quote currency.

5. Macro Headline & Sentiment Flow Alignment (Weight: ±2 Points):
   • +2: Live RSS headline narrative strongly supports asset direction.
   • -2: Live RSS headline narrative contradicts asset direction.

SCORE CONVICTION THRESHOLDS & MANDATORY OUTPUTS:
• Evaluate all 38 assets on the watchlist and generate TOP 3 TO 5 RANKED HIGH-CONVICTION TRADE SETUPS clearly labeled (*Setup #1* through *Setup #3* minimum, up to *Setup #5*).
• Score >= +7: HIGH-CONVICTION BULLISH (Eligible for Ranked Setup)
• Score <= -7: HIGH-CONVICTION BEARISH (Eligible for Ranked Setup)
• Score +3 to +6 / -3 to -6: MODERATE CONVICTION (Include in Directional Matrix only)
• Score -2 to +2: NO EDGE / NEUTRAL (Exclude from setups)

CONDENSED SETUP BREAKDOWN FORMAT (MANDATORY HIGH-DENSITY LAYOUT):
Every trade setup MUST follow this exact concise 3-bullet format:

🎯 *Setup #X: [ASSET] ([BUY / SELL])*
• *Score:* **[Composite Score] / +10** ([Conviction Rating]) | *Spot:* [Live Market Price]
• *Trigger & Invalidation:* Trigger: [1-sentence trigger] | Invalidation: [1-sentence zone]
• *Driver:* [1 concise sentence stating core fundamental catalyst based on provided data]
"""

PROMPT_TEMPLATES = {
    "ASIAN_OUTLOOK": f"""
Generate the **ASIAN & PACIFIC SESSION FUNDAMENTAL OUTLOOK** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

🌏 *ASIAN & PACIFIC SESSION FUNDAMENTAL OUTLOOK*
───────────────────────────

💵 *DEDICATED DXY & YEN MACRO ANALYSIS*
Concise analysis (2-3 sentences max) of DXY structure, Bank of Japan policy stance, Japanese Yields/JP225 correlation, and PBOC liquidity entering Asian trading hours.

───────────────────────────
📊 *COMPLETE WATCHLIST DIRECTIONAL BIAS MATRIX*
Categorize ALL 38 WATCHLIST ASSETS strictly into these 3 directional categories. Keep driver descriptions strictly to 1 short sentence per asset:

🟢 *BULLISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🔴 *BEARISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🟡 *NEUTRAL / NO EDGE:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

───────────────────────────
🔄 *OVERNIGHT FLOWS & SYDNEY/TOKYO LIQUIDITY SWEEPS*
1-2 sentence summary of Asian open liquidity, PBOC fixing, and Sydney/Tokyo price discovery.

───────────────────────────
🎯 *TOP HIGH-CONVICTION SETUPS (RANKED TOP 3 TO TOP 5)*
Provide 3 to 5 ranked setups matching the live price feed. You MUST use the mandatory condensed setup breakdown format.

───────────────────────────
⚠️ *RISK WARNING & LIQUIDITY NOTES (ASIAN SESSION EVENTS)*
1-2 sentence brief on upcoming RBA, RBNZ, BOJ announcements, Chinese prints, and Tokyo spread warnings.
""",

    "LONDON_OUTLOOK": f"""
Generate the **LONDON SESSION FUNDAMENTAL OUTLOOK** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

🏛 *LONDON SESSION FUNDAMENTAL OUTLOOK*
───────────────────────────

💵 *DEDICATED DXY MACRO ANALYSIS*
Concise analysis (2-3 sentences max) of DXY structure, Treasury yields, US rate expectations, and liquidity pools entering European hours.

───────────────────────────
📊 *COMPLETE WATCHLIST DIRECTIONAL BIAS MATRIX*
Categorize ALL 38 WATCHLIST ASSETS strictly into these 3 directional categories. Keep driver descriptions strictly to 1 short sentence per asset:

🟢 *BULLISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🔴 *BEARISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🟡 *NEUTRAL / NO EDGE:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

───────────────────────────
🔄 *INTRADAY CONTINUITY & THESIS CHECK*
1-2 sentence summary of prevailing macro narrative and adjustments from overnight Asian flows.

───────────────────────────
🎯 *TOP HIGH-CONVICTION SETUPS (RANKED TOP 3 TO TOP 5)*
Provide 3 to 5 ranked setups matching the live price feed. You MUST use the mandatory condensed setup breakdown format.

───────────────────────────
⚠️ *RISK WARNING & LIQUIDITY NOTES (HIGH IMPACT EVENTS TODAY)*
1-2 sentence brief on key European data releases, central bank speak, and spread warnings.
""",

    "NEW_YORK_OUTLOOK": f"""
Generate the **NEW YORK SESSION FUNDAMENTAL OUTLOOK** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

🏛 *NEW YORK SESSION FUNDAMENTAL OUTLOOK*
───────────────────────────

💵 *DEDICATED DXY MACRO ANALYSIS*
Concise analysis (2-3 sentences max) of US Dollar Index positioning heading into Wall Street open, FRED yield momentum, and Fed pricing.

───────────────────────────
📊 *COMPLETE WATCHLIST DIRECTIONAL BIAS MATRIX*
Categorize ALL 38 WATCHLIST ASSETS strictly into these 3 directional categories. Keep driver descriptions strictly to 1 short sentence per asset:

🟢 *BULLISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🔴 *BEARISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🟡 *NEUTRAL / NO EDGE:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

───────────────────────────
🔄 *INTRADAY CONTINUITY & THESIS TRANSITION*
1-2 sentence summary comparing London price action against the morning thesis and its impact on New York flows.

───────────────────────────
🎯 *TOP HIGH-CONVICTION SETUPS (RANKED TOP 3 TO TOP 5)*
Provide 3 to 5 ranked setups tailored for NY session volatility. You MUST use the mandatory condensed setup breakdown format.

───────────────────────────
⚠️ *RISK WARNING & LIQUIDITY NOTES (HIGH IMPACT EVENTS TODAY)*
1-2 sentence brief on upcoming US data releases, FOMC speakers, and volatility alerts.
""",

    "INTRADAY_UPDATE": f"""
Evaluate recent market price action and catalysts for an **INTRADAY UPDATE REPORT**.
Watchlist: {', '.join(WATCHLIST)}

INSTRUCTIONS:
You are provided with the text of the PREVIOUS SESSION REPORT below.
Compare the current live market pricing, yield delta, and latest news against that previous report.
First, determine if recent events or price action have caused a MATERIAL CHANGE in the market thesis, DXY stance, or setup validity.
- If NO material change: Begin output with `[NO_MATERIAL_CHANGE]` and explain why in 1 sentence.
- If YES (material change): Begin output with `[MATERIAL_CHANGE]` and generate the report below.

REQUIRED VISUAL LAYOUT (Only if [MATERIAL_CHANGE]):

🔄 *INTRADAY THESIS & BIAS SHIFT*
───────────────────────────

📌 *EVENT BREAKDOWN & MATERIAL SHIFT*
• *Previous Analysis Summary:* <1 short sentence recap referencing the provided prior briefing>
• *Catalyst / Market Movement:* <1 short sentence on observed price or data deviation>
• *Invalidations & Assumptions Broken:* <1 short sentence on previous setups that failed or hit invalidation>
• *New Core Market Thesis:* <1 concise sentence on updated directional bias>

───────────────────────────
💵 *DEDICATED DXY REACTION & YIELD IMPACT*
1-2 sentence summary of shift in DXY structure, FRED bond yields, and interest rate expectations.

───────────────────────────
📊 *UPDATED DIRECTIONAL BIAS MATRIX*
Categorize ALL 38 WATCHLIST ASSETS strictly into these 3 directional categories. Keep driver descriptions strictly to 1 short sentence per asset:

🟢 *BULLISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🔴 *BEARISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🟡 *NEUTRAL / NO EDGE:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

───────────────────────────
🎯 *UPDATED HIGH-CONVICTION SETUPS (RANKED TOP 3 TO TOP 5)*
Provide 3 to 5 updated setups reflecting the new thesis using live market prices. You MUST use the mandatory condensed setup breakdown format.

───────────────────────────
⚠️ *RISK WARNING & LIQUIDITY NOTES*
1-2 sentence brief on slippage, spread expansion, and session liquidity advisories.
""",

    "EOD_RECAP_WEEKDAY": f"""
Generate the **END OF DAY MACRO & FUNDAMENTAL RECAP** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

🌆 *END OF DAY MACRO & FUNDAMENTAL RECAP*
───────────────────────────

💵 *DEDICATED DXY & CROSS-ASSET RECAP*
Concise review (2-3 sentences max) of DXY closing performance across Asian, London, and NY sessions alongside Treasury yield behavior.

───────────────────────────
📊 *DAILY CLOSING BIAS MATRIX*
Categorize ALL 38 WATCHLIST ASSETS strictly into these 3 directional categories. Keep driver descriptions strictly to 1 short sentence per asset:

🟢 *BULLISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🔴 *BEARISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🟡 *NEUTRAL / NO EDGE:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

───────────────────────────
🎯 *SETUP VERDICT SCOREBOARD (SESSION EXECUTION AUDIT)*
Concise evaluation of execution quality and outcomes for today's setups:
• *London Session Setups:* [Grade A to F | Brief R:R outcome]
• *New York Session Setups:* [Grade A to F | Brief R:R outcome]

───────────────────────────
🔄 *INTRADAY DEVELOPMENTS & NARRATIVE SHIFT*
1-2 sentence summary of key market drivers, unexpected decoupling events, and HTF structural shifts.

───────────────────────────
📅 *TOMORROW'S CALENDAR & OVERNIGHT WATCHLIST*
1-2 sentence brief on upcoming high-impact events and daily pivot levels.

───────────────────────────
💡 *INSTITUTIONAL EDGE & PROCESS TAKEAWAY*
Single highest-leverage execution or risk discipline lesson derived from today's price action.
""",

    "EOD_RECAP_FRIDAY": f"""
Generate the **WEEKEND WRAP & SUNDAY OPEN MACRO OUTLOOK** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

🌆 *WEEKEND WRAP & SUNDAY OPEN MACRO OUTLOOK*
───────────────────────────

💵 *WEEKLY DXY & CROSS-ASSET CLOSING RECAP*
Concise review (2-3 sentences max) of weekly DXY performance, Friday NY close structure, and Treasury yield shifts.

───────────────────────────
📊 *WEEKLY CLOSING BIAS MATRIX*
Categorize ALL 38 WATCHLIST ASSETS strictly into these 3 directional categories. Keep driver descriptions strictly to 1 short sentence per asset:

🟢 *BULLISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🔴 *BEARISH BIAS:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

🟡 *NEUTRAL / NO EDGE:*
• *[ASSET]* (Spot: [Price]) | [1 short sentence driver]

───────────────────────────
🎯 *WEEKLY SETUP VERDICT SCOREBOARD*
Concise audit of execution discipline and trade setup outcomes across the entire trading week (Grades A to F).

───────────────────────────
🔄 *WEEKLY STRUCTURAL SHIFTS & OPEN IMBALANCES*
1-2 sentence summary of HTF weekly/daily Market Structure Shifts (CHoCH/BOS) and major unmitigated FVGs carried into next week.

───────────────────────────
📅 *UPCOMING WEEK CALENDAR & SUNDAY OPEN OUTLOOK*
1-2 sentence brief on upcoming high-impact events and preliminary directional bias for Sunday open.

───────────────────────────
💡 *WEEKLY PROCESS LESSON & INSTITUTIONAL EDGE*
Highest-leverage risk management or execution takeaway to refine for next week's trading desk.
"""
}

# ---------------------------------------------------------------------------
# Market Data, FRED, Calendar & News Fetchers
# ---------------------------------------------------------------------------
def fetch_fred_yields() -> str:
    """Pulls authoritative US Treasury yield series from FRED API with logging."""
    if not FRED_API_KEY:
        return "FRED API Key not configured. US Treasury Yield benchmark unavailable."

    logging.info("Fetching US Treasury Yield series from FRED API...")
    try:
        url_10y = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={FRED_API_KEY}&file_type=json&sort_order=desc&limit=2"
        res = requests.get(url_10y, timeout=5)
        if res.status_code == 200:
            obs = res.json().get("observations", [])
            if len(obs) >= 2 and obs[0]["value"] != ".":
                latest_10y = float(obs[0]["value"])
                prev_10y = float(obs[1]["value"])
                delta_10y = latest_10y - prev_10y
                return f"FRED US 10Y Yield: {latest_10y:.2f}% (24h Change: {delta_10y:+.2f}%)"
    except Exception as e:
        logging.warning(f"FRED API fetch exception: {sanitize_error_text(str(e))}")

    return "FRED Yield Feed: Data temporarily unavailable."

def compute_copper_gold_ratio() -> str:
    """Computes real-time Copper/Gold ratio as a leading risk indicator."""
    try:
        copper = yf.Ticker("HG=F")
        gold = yf.Ticker("XAUUSD=X")

        c_price = copper.fast_info.get("lastPrice") or copper.fast_info.get("regularMarketPrice")
        g_price = gold.fast_info.get("lastPrice") or gold.fast_info.get("regularMarketPrice")

        if c_price and g_price and g_price > 0:
            ratio = (c_price / g_price) * 1000
            return f"Copper/Gold Intermarket Ratio: {ratio:.4f} (HG: {c_price:.4f} | XAU: {g_price:.2f})"
    except Exception as e:
        logging.warning(f"Could not compute Copper/Gold ratio: {e}")

    return "Copper/Gold Intermarket Ratio: Data unavailable"

def fetch_live_market_data() -> str:
    ticker_mapping = {
        # Majors (7)
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "NZDUSD": "NZDUSD=X",
        "AUDUSD": "AUDUSD=X", "USDCHF": "USDCHF=X", "USDCAD": "USDCAD=X", "USDJPY": "USDJPY=X",
        # EUR Crosses (6)
        "EURGBP": "EURGBP=X", "EURAUD": "EURAUD=X", "EURNZD": "EURNZD=X",
        "EURCAD": "EURCAD=X", "EURCHF": "EURCHF=X", "EURJPY": "EURJPY=X",
        # GBP Crosses (5)
        "GBPAUD": "GBPAUD=X", "GBPNZD": "GBPNZD=X", "GBPCAD": "GBPCAD=X",
        "GBPCHF": "GBPCHF=X", "GBPJPY": "GBPJPY=X",
        # AUD / NZD Crosses (7)
        "AUDNZD": "AUDNZD=X", "AUDCAD": "AUDCAD=X", "AUDCHF": "AUDCHF=X", "AUDJPY": "AUDJPY=X",
        "NZDCAD": "NZDCAD=X", "NZDCHF": "NZDCHF=X", "NZDJPY": "NZDJPY=X",
        # CAD / CHF Crosses (3)
        "CADCHF": "CADCHF=X", "CADJPY": "CADJPY=X", "CHFJPY": "CHFJPY=X",
        # Commodities, Crypto & Equity Indices
        "GOLD": "XAUUSD=X", "SILVER": "XAGUSD=X", "COPPER": "HG=F",
        "SP500": "^GSPC", "NAS100": "^NDX", "GER40": "^GDAXI", "JP225": "^N225",
        "DXY": "DX-Y.NYB", "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD",
        # Yield Benchmarks
        "US10Y_YIELD": "^TNX", "US02Y_YIELD": "^IRX", "VIX_VOLATILITY": "^VIX"
    }

    logging.info("Fetching real-time spot market prices via fast_info live feed...")
    price_lines = []

    for asset, symbol in ticker_mapping.items():
        try:
            ticker = yf.Ticker(symbol)
            last_price = ticker.fast_info.get("lastPrice") or ticker.fast_info.get("regularMarketPrice")
            prev_close = ticker.fast_info.get("previousClose") or ticker.fast_info.get("regularMarketPreviousClose")

            if last_price is None or prev_close is None:
                hist = ticker.history(period="2d", interval="1m")
                if not hist.empty:
                    last_price = hist['Close'].iloc[-1]
                    hist_daily = ticker.history(period="5d", interval="1d")
                    prev_close = hist_daily['Close'].iloc[-2] if len(hist_daily) >= 2 else last_price

            if last_price is not None and prev_close is not None:
                pct_change = ((last_price - prev_close) / prev_close) * 100
                price_fmt = f"{last_price:.4f}" if last_price < 10 else f"{last_price:.2f}"
                change_sign = "+" if pct_change >= 0 else ""
                price_lines.append(f"- {asset}: {price_fmt} ({change_sign}{pct_change:.2f}% 24h)")
            else:
                price_lines.append(f"- {asset}: Data unavailable")

        except Exception as e:
            logging.warning(f"Could not extract live price for {asset}: {e}")
            price_lines.append(f"- {asset}: N/A")

    return "\n".join(price_lines)

def fetch_live_macro_headlines() -> str:
    logging.info("Fetching real-time macro headlines via RSS feed...")
    rss_url = "https://news.google.com/rss/search?q=forex+central+bank+inflation+fed+ecb+boj+when:1d&hl=en-US&gl=US&ceid=US:en"
    
    try:
        res = requests.get(rss_url, timeout=10)
        if res.status_code != 200:
            return "No recent headline feed returned."
            
        root = ET.fromstring(res.content)
        headlines = []
        for item in root.findall('./channel/item')[:6]:
            title = item.find('title').text if item.find('title') is not None else ""
            if title:
                clean_title = re.sub(r"\s*-\s*[^-]+$", "", title)
                headlines.append(f"• {clean_title}")
                
        return "\n".join(headlines) if headlines else "No major breaking macro headlines found."
    except Exception as e:
        logging.warning(f"Could not fetch macro news RSS: {e}")
        return "Headline feed temporarily offline."

def fetch_economic_calendar() -> str:
    if not FINNHUB_API_KEY:
        return "Finnhub API key not configured. Skipping structured economic calendar feed."

    logging.info("Fetching economic calendar data from Finnhub API...")
    today_str = time.strftime("%Y-%m-%d", time.gmtime())
    url = f"https://finnhub.io/api/v1/calendar/economic?from={today_str}&to={today_str}&token={FINNHUB_API_KEY}"

    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return "Economic calendar API returned non-200 status code."
        
        events = res.json().get("economicCalendar", [])
        if not events:
            return "No economic calendar releases found for today."

        filtered_events = [e for e in events if str(e.get("impact", "")).lower() in ["high", "medium", "3", "2"]]
        
        lines = []
        for e in filtered_events[:10]:
            event_name = e.get("event", "Economic Release")
            country = e.get("country", "GLOBAL")
            actual = e.get("actual", "Pending")
            estimate = e.get("estimate", "N/A")
            prev = e.get("prev", "N/A")
            lines.append(f"• [{country}] {event_name} | Actual: {actual} | Forecast: {estimate} | Previous: {prev}")

        return "\n".join(lines) if lines else "No high/medium impact calendar releases scheduled today."
    except Exception as e:
        logging.warning(f"Could not fetch economic calendar: {sanitize_error_text(str(e))}")
        return "Economic calendar fetch failed."

# ---------------------------------------------------------------------------
# Visual Banner Generator
# ---------------------------------------------------------------------------
def generate_session_banner(mode: str) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=150)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    display_titles = {
        "ASIAN_OUTLOOK": "ASIAN & PACIFIC SESSION OUTLOOK",
        "LONDON_OUTLOOK": "LONDON SESSION OUTLOOK",
        "NEW_YORK_OUTLOOK": "NEW YORK SESSION OUTLOOK",
        "INTRADAY_UPDATE": "INTRADAY THESIS UPDATE",
        "EOD_RECAP": "END-OF-DAY MACRO RECAP"
    }
    subtitle = display_titles.get(mode, "MARKET INTELLIGENCE BRIEF")

    # Map session state directly to report mode to ensure accurate pre-session tagging
    state_mapping = {
        "ASIAN_OUTLOOK": "PRE-MARKET: SYDNEY / TOKYO OPEN PREPARATION",
        "LONDON_OUTLOOK": "PRE-MARKET: FRANKFURT / LONDON OPEN PREPARATION",
        "NEW_YORK_OUTLOOK": "PRE-MARKET: NEW YORK / WALL STREET OPEN PREPARATION",
        "INTRADAY_UPDATE": "MID-SESSION: INTRADAY LIQUIDITY & THESIS AUDIT",
        "EOD_RECAP": "MARKET WRAP: DAILY CLOSING BENCHMARKS & REVIEW"
    }
    session_state = state_mapping.get(mode, "ACTIVE MARKET INTELLIGENCE")

    ax.text(0.05, 0.75, "CASCADE: NESTED CONFLUENCE TRADING SYSTEM", fontsize=10, color='#8b949e', fontweight='bold', fontfamily='monospace')
    ax.text(0.05, 0.48, subtitle, fontsize=20, color='#ffffff', fontweight='bold', fontfamily='sans-serif')
    ax.text(0.05, 0.28, session_state, fontsize=8, color='#3fb950', fontweight='bold', fontfamily='monospace')
    
    timestamp_str = time.strftime("%A, %d %b %Y | %H:%M UTC", time.gmtime())
    ax.text(0.05, 0.12, f"GENERATED: {timestamp_str}", fontsize=9, color='#58a6ff', fontweight='bold', fontfamily='monospace')

    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

# ---------------------------------------------------------------------------
# Formatting, Security, Logging & Telegram Helpers
# ---------------------------------------------------------------------------
def sanitize_error_text(text: str) -> str:
    """Masks secrets and query parameters from log messages and Telegram error notifications."""
    text = re.sub(r"(api_key|token|key|bot[0-9]+:)[=:][A-Za-z0-9_\-]+", r"\1=[REDACTED]", text, flags=re.IGNORECASE)
    return text

def sanitize_telegram_markdown(text: str) -> str:
    """Balances unclosed bold and italic tags to protect against Telegram Markdown parse rejection."""
    if text.count("*") % 2 != 0:
        text += "*"
    if text.count("_") % 2 != 0:
        text += "_"
    return text

def chunk_telegram_text(text: str, max_bytes: int = 3900) -> list[str]:
    lines = text.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_bytes:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def log_setups_to_csv(report_text: str, mode: str):
    file_path = "setups_log.csv"
    file_exists = os.path.isfile(file_path)
    timestamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    
    setup_matches = re.findall(r"🎯\s*\*Setup\s*#\d+:.*?\*", report_text)
    
    if not setup_matches:
        logging.info("No formatted trade setups detected to write into CSV log.")
        return

    try:
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "System", "Session Mode", "Setup Details"])
            
            for setup in setup_matches:
                clean_title = setup.replace("*", "").replace("🎯", "").strip()
                writer.writerow([timestamp, "CASCADE", mode, clean_title])
        logging.info(f"Successfully recorded {len(setup_matches)} setup(s) into {file_path}")
    except Exception as e:
        logging.warning(f"Could not log setups to CSV file: {e}")

def send_telegram_media_and_message(text: str, banner_buf: io.BytesIO) -> bool:
    photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    msg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # 1. Send Visual Banner Image
    try:
        files = {'photo': ('banner.png', banner_buf.getvalue(), 'image/png')}
        photo_res = requests.post(
            photo_url, 
            data={'chat_id': TELEGRAM_CHANNEL_ID, 'caption': "📊 *CASCADE: Intelligence Briefing*", 'parse_mode': 'Markdown'}, 
            files=files, 
            timeout=15
        )
        if photo_res.status_code != 200:
            logging.warning(f"Banner upload alert ({photo_res.status_code}): {photo_res.text}")
    except Exception as e:
        logging.warning(f"Banner photo dispatch exception: {sanitize_error_text(str(e))}")

    # 2. Sanitize and Send Chunked Text Report
    sanitized_text = sanitize_telegram_markdown(text)
    chunks = chunk_telegram_text(sanitized_text)

    success = True
    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "text": chunk,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        res = requests.post(msg_url, json=payload, timeout=15)
        
        # Fallback to plain text if Markdown parsing fails
        if res.status_code != 200:
            logging.warning(f"Markdown dispatch failed ({res.status_code}). Attempting plain text fallback...")
            payload["parse_mode"] = ""
            res_fallback = requests.post(msg_url, json=payload, timeout=15)
            if res_fallback.status_code != 200:
                logging.error(f"Telegram API error: {res_fallback.text}")
                success = False

    return success

def send_admin_alert(message: str):
    if not TELEGRAM_ADMIN_CHAT_ID or not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    clean_msg = sanitize_error_text(message)
    payload = {"chat_id": TELEGRAM_ADMIN_CHAT_ID, "text": f"⚠️ *SYSTEM ALERT*\n{clean_msg}", "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logging.error(f"Admin alert dispatch failed: {e}")

def count_setups(report_text: str) -> int:
    """Strictly matches standardized setup headers to prevent false positives."""
    return len(re.findall(r"🎯\s*\*Setup\s*#\d+:", report_text))

# ---------------------------------------------------------------------------
# Report Generation & Logic Pipeline
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Forex Fundamental Analyser")
    parser.add_argument("positional_mode", nargs="?", default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--feed", type=str, default=None)
    args, _ = parser.parse_known_args()

    raw_mode = args.mode or args.positional_mode or "LONDON_OUTLOOK"
    cleaned = raw_mode.strip().upper()

    mapping = {
        "ASIA": "ASIAN_OUTLOOK", "ASIAN": "ASIAN_OUTLOOK", "ASIAN_OUTLOOK": "ASIAN_OUTLOOK",
        "LONDON": "LONDON_OUTLOOK", "LONDON_OUTLOOK": "LONDON_OUTLOOK",
        "NEW_YORK": "NEW_YORK_OUTLOOK", "NEW_YORK_OUTLOOK": "NEW_YORK_OUTLOOK", "NY": "NEW_YORK_OUTLOOK",
        "INTRADAY": "INTRADAY_UPDATE", "INTRADAY_UPDATE": "INTRADAY_UPDATE",
        "POST_NEWS": "INTRADAY_UPDATE", "POST_NEWS_UPDATE": "INTRADAY_UPDATE",
        "EOD": "EOD_RECAP", "EOD_RECAP": "EOD_RECAP"
    }

    return mapping.get(cleaned, "LONDON_OUTLOOK"), args.feed

def validate_environment():
    missing = []
    if not GEMINI_API_KEY: missing.append("GEMINI_API_KEY")
    if not TELEGRAM_BOT_TOKEN: missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHANNEL_ID: missing.append("TELEGRAM_CHANNEL_ID")
    if not FRED_API_KEY: missing.append("FRED_API_KEY")

    if missing:
        logging.error(f"Missing required environment variable(s): {', '.join(missing)}")
        sys.exit(1)

def generate_report_with_retry(client: genai.Client, mode: str, custom_feed: str = None, max_retries: int = 3) -> str:
    # Handle Friday EOD Market Close vs Weekday EOD
    if mode == "EOD_RECAP":
        is_friday = (time.gmtime().tm_wday == 4)
        active_template_key = "EOD_RECAP_FRIDAY" if is_friday else "EOD_RECAP_WEEKDAY"
    else:
        active_template_key = mode

    prompt = PROMPT_TEMPLATES[active_template_key]
    
    # State Memory: Inject previous report if executing INTRADAY_UPDATE
    if mode == "INTRADAY_UPDATE":
        if os.path.isfile(LAST_REPORT_CACHE_FILE):
            try:
                with open(LAST_REPORT_CACHE_FILE, "r", encoding="utf-8") as f:
                    prev_report = f.read().strip()
                if prev_report:
                    prompt += f"\n\nPREVIOUS SESSION REPORT FOR CONTINUITY AUDIT:\n{prev_report}"
            except Exception as e:
                logging.warning(f"Could not load previous report cache: {e}")
        else:
            prompt += "\n\nPREVIOUS SESSION REPORT FOR CONTINUITY AUDIT:\n[Initial session run. No previous report cached on disk.]"

    live_prices = fetch_live_market_data()
    fred_yields = fetch_fred_yields()
    copper_gold_ratio = compute_copper_gold_ratio()
    live_headlines = fetch_live_macro_headlines()
    economic_calendar = fetch_economic_calendar()

    prompt += f"\n\nLIVE MARKET PRICE & YIELD FEED (MANDATORY REFERENCE FOR ALL PRICES):\n{live_prices}"
    prompt += f"\n\nFRED MACRO YIELD DATA:\n{fred_yields}"
    prompt += f"\n\nINTERMARKET RISK METRIC:\n{copper_gold_ratio}"
    prompt += f"\n\nSTRUCTURED ECONOMIC CALENDAR (ACTUAL VS FORECAST):\n{economic_calendar}"
    prompt += f"\n\nBREAKING MACRO HEADLINES (24H):\n{live_headlines}"

    if custom_feed:
        prompt += f"\n\nCUSTOM BREAKING EVENT FEED:\n{custom_feed}"

    # Model Fallback Chain to protect against Google server capacity spikes (503 UNAVAILABLE)
    model_fallback_chain = [
        PREFERRED_MODEL,
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash"
    ]

    report_text = ""
    setup_count = 0

    for model_name in model_fallback_chain:
        logging.info(f"Attempting report generation with model target: '{model_name}'...")
        
        for attempt in range(1, max_retries + 1):
            logging.info(f"Calling Gemini ({model_name}) for mode '{active_template_key}' (Attempt {attempt}/{max_retries})...")

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTIONS,
                        temperature=0.3,
                    )
                )
                report_text = (response.text or "").strip()
                
                # Check for Intraday No Material Change flag
                if "[NO_MATERIAL_CHANGE]" in report_text:
                    return "[NO_MATERIAL_CHANGE]"

                if not report_text.endswith(MANDATORY_DISCLAIMER.strip()):
                    report_text += f"\n\n{MANDATORY_DISCLAIMER.strip()}"

                if mode == "EOD_RECAP":
                    return report_text

                setup_count = count_setups(report_text)
                logging.info(f"Detected {setup_count} trade setup(s) in output.")

                if setup_count >= 3:
                    return report_text

                logging.warning(f"Setup count validation failed ({setup_count}/3 required). Retrying prompt...")
                prompt += "\n\nCRITICAL REQUIREMENT MISSING: You MUST generate AT LEAST 3 to 5 ranked, high-conviction trade setups clearly labeled (*Setup #1*, *Setup #2*, etc.) using the exact live prices provided and following the condensed setup breakdown format."

            except Exception as e:
                error_str = str(e)
                logging.error(f"API Error with model '{model_name}': {sanitize_error_text(error_str)}")
                
                # Handle Google server-side capacity spikes (503 UNAVAILABLE)
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    if attempt < max_retries:
                        sleep_duration = 4 * attempt
                        logging.info(f"503 High Demand detected on Google servers. Waiting {sleep_duration}s...")
                        time.sleep(sleep_duration)
                    else:
                        logging.warning(f"Model '{model_name}' exhausted retries due to 503 capacity limits. Switching to next model in fallback chain...")
                        break
                else:
                    if attempt < max_retries:
                        time.sleep(3)
                    else:
                        break

        if report_text and setup_count >= 3:
            return report_text

    logging.warning("All fallback models exhausted. Dispatching best available output.")
    send_admin_alert(f"Mode `{mode}` encountered server capacity or output threshold limits across all fallback models.")
    return report_text

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    validate_environment()
    mode, custom_feed = parse_arguments()

    logging.info(f"Starting pipeline | Mode: {mode} | Primary Model: {PREFERRED_MODEL}")

    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        report = generate_report_with_retry(client=client, mode=mode, custom_feed=custom_feed)

        # Gatekeeper check for suppressed intraday updates
        if report == "[NO_MATERIAL_CHANGE]":
            logging.info("Intraday reassessment completed: No material shift in macro thesis or setups. Telegram update suppressed.")
            sys.exit(0)

        # Clean tag if present
        report = report.replace("[MATERIAL_CHANGE]", "").strip()

        # Operational Guard: Prevent publishing empty or blank text reports
        if not report or not report.strip():
            logging.error("Generated report is completely empty. Aborting dispatch.")
            send_admin_alert(f"Execution failed: Report generation produced blank output for mode `{mode}`. Telegram post suppressed.")
            sys.exit(1)

        # Update disk cache for subsequent intraday comparisons
        try:
            with open(LAST_REPORT_CACHE_FILE, "w", encoding="utf-8") as f:
                f.write(report)
        except Exception as e:
            logging.warning(f"Could not persist report to cache file: {e}")

        # Log setups to CSV file
        log_setups_to_csv(report, mode)

        logging.info("Report generation complete. Generating visual session banner...")
        banner_buf = generate_session_banner(mode)

        logging.info("Dispatching visual banner and text report to Telegram...")
        if send_telegram_media_and_message(report, banner_buf):
            logging.info("Successfully published session intelligence package to Telegram.")
        else:
            logging.error("Failed to post report package to Telegram.")
            sys.exit(1)

    except Exception as e:
        error_msg = f"Pipeline execution failed for mode `{mode}`:\n`{sanitize_error_text(str(e))}`"
        logging.error(error_msg, exc_info=True)
        send_admin_alert(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
