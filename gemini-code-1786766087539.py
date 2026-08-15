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

PREFERRED_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
TELEGRAM_ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY", "d9cca9a169f732432e312ad7efcc4fae")

MANDATORY_DISCLAIMER = """───────────────────────────
⚠️ *RISK DISCLAIMER & SUBSCRIBER NOTICE*
_The setups and directional biases provided above are analytical insights for educational purposes only. Do NOT blindly copy or execute these ideas. Trading requires independent trade confirmation, strict risk management, appropriate position sizing, and adherence to stated invalidation levels._"""

# ---------------------------------------------------------------------------
# System Instructions & Telegram Formatting Rules
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = f"""
You are a Lead Institutional Macro & Forex Fundamental Analyst operating the Apex Macro Desk decision engine.
You analyze global financial markets across this complete watchlist:
{', '.join(WATCHLIST)}.

TELEGRAM VISUAL FORMATTING RULES (CRITICAL):
1. DO NOT use `#` or `##` Markdown headers (Telegram displays them as raw text).
2. Use Emoji + Bold ALL-CAPS for section headers (e.g., `📌 *DEDICATED DXY MACRO ANALYSIS*`).
3. Use `───────────────────────────` as a divider line between major sections.
4. Use `•` bullet points with bold field names for structured data (e.g., `• *Macro Catalyst:* CPI beats consensus`).
5. Keep paragraphs short (2-3 sentences max) with clean line breaks for effortless mobile reading.

INSTITUTIONAL WEIGHTED SCORECARD (MANDATORY EVALUATION ENGINE):
For every pair on the watchlist, calculate a Composite Edge Score from -10 to +10 by evaluating these 6 weighted factors:

1. Yield Differential Momentum (Weight: ±2 Points):
   • +2: Base currency bond yields outpace Quote currency yields over 24h.
   • -2: Quote currency bond yields outpace Base currency yields over 24h.

2. DXY / Relative Strength Divergence (Weight: ±1 Point):
   • +1: Asset rises while DXY is strong (Class-A Relative Strength).
   • -1: Asset falls while DXY is weak (Class-A Relative Weakness).

3. Intermarket Risk Regime Alignment (Weight: ±1 Point):
   • +1: Copper/Gold ratio rising + SP500/JP225 up -> Aligned with Long Growth currencies (AUD, NZD, CAD).
   • -1: Copper/Gold ratio falling + VIX rising -> Aligned with Safe Havens (JPY, CHF, USD).

4. Institutional COT Positioning Hierarchy (Weight: ±3 / ±2 / ±1 Points):
   • ±3: Asset Managers heavily net positioning alignment.
   • ±2: Leveraged Funds net positioning alignment.
   • ±1: Non-Commercial net positioning alignment.

5. Retail Sentiment Contrarian Flag (Weight: ±1 Point):
   • +1: Retail sentiment is heavily short (>70% short) -> Bullish contrarian bias.
   • -1: Retail sentiment is heavily long (>70% long) -> Bearish contrarian bias.

6. Triangular Leg Agreement Modifier (Weight: ±1 Point):
   • +1: Both underlying USD major legs push in the same direction.
   • 0: USD major legs diverge (Do NOT eliminate the setup; assign 0 bonus points).

SCORE CONVICTION THRESHOLDS & MANDATORY OUTPUTS:
• You MUST evaluate all 38 assets on the watchlist and generate AT LEAST 10 RANKED HIGH-CONVICTION TRADE SETUPS clearly labeled (*Setup #1* through *Setup #10*) for session reports.
• Score >= +6: HIGH-CONVICTION BULLISH (Eligible for Ranked Setup)
• Score <= -6: HIGH-CONVICTION BEARISH (Eligible for Ranked Setup)
• Score +3 to +5 / -3 to -5: MODERATE CONVICTION (Include in Directional Matrix only)
• Score -2 to +2: NO EDGE / NEUTRAL (Exclude from setups)

CONDENSED SETUP BREAKDOWN FORMAT (MANDATORY):
Every trade setup MUST follow this exact high-density condensed format:

🎯 *Setup #X: [ASSET] ([BUY / SELL])*
• *Score:* **[Composite Score] / +10** ([Conviction Rating]) | *Spot:* [Live Market Price]
• *Score Breakdown:* `Yield:[+2/-2]` | `DXY:[+1/-1]` | `Risk:[+1/-1]` | `COT:[+3/+2/+1]` | `Sent:[+1/-1]` | `Leg:[+1/0]`
• *Execution:* **Entry:** [Price] | **Target:** [Price] | **Stop:** [Price]
• *Key Driver:* [1-2 sentence core fundamental macro driver and catalyst]
• *Timing:* [Session timing and execution window]
"""

PROMPT_TEMPLATES = {
    "ASIAN_OUTLOOK": f"""
Generate the **ASIAN & PACIFIC SESSION FUNDAMENTAL OUTLOOK** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

🌏 *ASIAN & PACIFIC SESSION FUNDAMENTAL OUTLOOK*
───────────────────────────

💵 *DEDICATED DXY & YEN MACRO ANALYSIS*
Deep dive into DXY structure, Bank of Japan policy stance, Japanese Yields/JP225 correlation, and PBOC liquidity injections entering Asian trading hours.

───────────────────────────
📊 *COMPLETE WATCHLIST DIRECTIONAL BIAS MATRIX*
Explicit Directional Bias (🟢 Bullish / 🔴 Bearish / 🟡 Neutral) and 1-sentence macro driver for ALL WATCHLIST ASSETS:
{', '.join(WATCHLIST)}.

───────────────────────────
🔄 *OVERNIGHT FLOWS & SYDNEY/TOKYO LIQUIDITY SWEEPS*
Analysis of Asian market open liquidity, PBOC fixing, and Sydney/Tokyo price discovery.

───────────────────────────
🎯 *TOP HIGH-CONVICTION SETUPS (RANKED #1 TO #10 MINIMUM)*
Provide AT LEAST 10 ranked setups matching the live price data feed across FX majors, crosses, and commodities. You MUST use the mandatory condensed setup breakdown format.

───────────────────────────
⚠️ *RISK WARNING & LIQUIDITY NOTES (ASIAN SESSION EVENTS)*
Upcoming RBA, RBNZ, BOJ announcements, Chinese economic prints, and Tokyo spread warnings.
""",

    "LONDON_OUTLOOK": f"""
Generate the **LONDON SESSION FUNDAMENTAL OUTLOOK** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

🏛 *LONDON SESSION FUNDAMENTAL OUTLOOK*
───────────────────────────

💵 *DEDICATED DXY MACRO ANALYSIS*
Deep dive into DXY structure, Treasury yields, US rate expectations, and liquidity pools entering European hours. Clarify non-mechanical currency divergences.

───────────────────────────
📊 *COMPLETE WATCHLIST DIRECTIONAL BIAS MATRIX*
Explicit Directional Bias (🟢 Bullish / 🔴 Bearish / 🟡 Neutral) and 1-sentence macro driver for ALL WATCHLIST ASSETS:
{', '.join(WATCHLIST)}.

───────────────────────────
🔄 *INTRADAY CONTINUITY & THESIS CHECK*
Prevailing macro narrative and adjustments from overnight Asian flows.

───────────────────────────
🎯 *TOP HIGH-CONVICTION SETUPS (RANKED #1 TO #10 MINIMUM)*
Provide AT LEAST 10 ranked setups matching the live price data feed. You MUST use the mandatory condensed setup breakdown format.

───────────────────────────
⚠️ *RISK WARNING & LIQUIDITY NOTES (HIGH IMPACT EVENTS TODAY)*
Key European data releases, central bank speak, economic calendar catalysts, and spread warnings.
""",

    "NEW_YORK_OUTLOOK": f"""
Generate the **NEW YORK SESSION FUNDAMENTAL OUTLOOK** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

🏛 *NEW YORK SESSION FUNDAMENTAL OUTLOOK*
───────────────────────────

💵 *DEDICATED DXY MACRO ANALYSIS*
US Dollar Index positioning heading into Wall Street open, FRED yield momentum, and Fed pricing.

───────────────────────────
📊 *COMPLETE WATCHLIST DIRECTIONAL BIAS MATRIX*
Explicit Directional Bias (🟢 Bullish / 🔴 Bearish / 🟡 Neutral) for ALL WATCHLIST ASSETS:
{', '.join(WATCHLIST)}.

───────────────────────────
🔄 *INTRADAY CONTINUITY & THESIS TRANSITION*
Compare London price action against the morning thesis. Explicitly state: What changed during London, why it changed, and how it impacts New York flows.

───────────────────────────
🎯 *TOP HIGH-CONVICTION SETUPS (RANKED #1 TO #10 MINIMUM)*
AT LEAST 10 ranked setups tailored for NY session volatility using live asset pricing. You MUST use the mandatory condensed setup breakdown format.

───────────────────────────
⚠️ *RISK WARNING & LIQUIDITY NOTES (HIGH IMPACT EVENTS TODAY)*
Upcoming US data releases, FOMC speakers, and volatility alerts.
""",

    "INTRADAY_UPDATE": f"""
Evaluate recent market price action and catalysts for an **INTRADAY UPDATE REPORT**.
Watchlist: {', '.join(WATCHLIST)}

INSTRUCTIONS:
First, determine if recent events or price action have caused a MATERIAL CHANGE in the market thesis, DXY stance, or setup validity.
- If NO material change: Begin output with `[NO_MATERIAL_CHANGE]` and explain why.
- If YES (material change): Begin output with `[MATERIAL_CHANGE]` and generate the report below.

REQUIRED VISUAL LAYOUT (Only if [MATERIAL_CHANGE]):

🔄 *INTRADAY THESIS & BIAS SHIFT*
───────────────────────────

📌 *EVENT BREAKDOWN & MATERIAL SHIFT*
• *Previous Analysis Summary:* <Brief recap of earlier bias>
• *Catalyst / Market Movement:* <What occurred>
• *Invalidations & Assumptions Broken:* <Which biases/setups are now dead>
• *New Core Market Thesis:* <Updated macro stance>

───────────────────────────
💵 *DEDICATED DXY REACTION & YIELD IMPACT*
Shift in DXY structure, FRED bond yields, and interest rate expectations.

───────────────────────────
📊 *UPDATED DIRECTIONAL BIAS MATRIX*
Post-event bias for ALL WATCHLIST ASSETS: {', '.join(WATCHLIST)}.

───────────────────────────
🎯 *UPDATED HIGH-CONVICTION SETUPS (RANKED #1 TO #10 MINIMUM)*
AT LEAST 10 updated setups reflecting the new thesis using live market prices. You MUST use the mandatory condensed setup breakdown format.

───────────────────────────
⚠️ *RISK WARNING & LIQUIDITY NOTES*
Slippage, spread expansion, and session liquidity advisories.
""",

    "EOD_RECAP_WEEKDAY": f"""
Generate the **END OF DAY MACRO & FUNDAMENTAL RECAP** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

🌆 *END OF DAY MACRO & FUNDAMENTAL RECAP*
───────────────────────────

💵 *DEDICATED DXY & CROSS-ASSET RECAP*
In-depth review of DXY closing performance across Asian, London, and NY sessions alongside Treasury yield behavior and intermarket flows.

───────────────────────────
📊 *DAILY CLOSING BIAS MATRIX*
Closing price action and directional bias (🟢 Bullish / 🔴 Bearish / 🟡 Neutral) for ALL WATCHLIST ASSETS:
{', '.join(WATCHLIST)}.

───────────────────────────
🎯 *SETUP VERDICT SCOREBOARD (SESSION EXECUTION AUDIT)*
Review execution and outcomes for today's London and New York session setups:
• *London Session Setups:* [Evaluate execution quality, assign Execution Grade (A to F), and state R:R result]
• *New York Session Setups:* [Evaluate execution quality, assign Execution Grade (A to F), and state R:R result]

───────────────────────────
🔄 *INTRADAY DEVELOPMENTS & NARRATIVE SHIFT*
Summary of key market drivers, unexpected decoupling events, HTF structural shifts (CHoCH/BOS), and unmitigated Fair Value Gaps (FVGs) left open for tomorrow.

───────────────────────────
📅 *TOMORROW'S CALENDAR & OVERNIGHT WATCHLIST*
Key high-impact events for upcoming sessions, critical daily pivot levels, and preliminary directional outlook.

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
Comprehensive review of weekly DXY performance, Friday NY close structure, and Treasury yield shifts across all global sessions.

───────────────────────────
📊 *WEEKLY CLOSING BIAS MATRIX*
Weekly closing directional state (🟢 Bullish / 🔴 Bearish / 🟡 Neutral) for ALL WATCHLIST ASSETS:
{', '.join(WATCHLIST)}.

───────────────────────────
🎯 *WEEKLY SETUP VERDICT SCOREBOARD*
Audit of execution discipline and trade setup outcomes across the entire trading week (Grades A to F).

───────────────────────────
🔄 *WEEKLY STRUCTURAL SHIFTS & OPEN IMBALANCES*
HTF weekly/daily Market Structure Shifts (CHoCH/BOS) and major unmitigated Fair Value Gaps (FVGs) carried into next week.

───────────────────────────
📅 *UPCOMING WEEK CALENDAR & SUNDAY OPEN OUTLOOK*
Key high-impact events scheduled for the coming week, weekend geopolitical risk factors, and preliminary directional bias for Sunday market open / Monday Asia session.

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
        return "FRED API Key missing. Yield data defaulting to Yahoo Finance feed."

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
        logging.warning(f"FRED API fetch exception: {e}")

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
            
            # Fetch live real-time price and previous close from ticker fast_info
            last_price = ticker.fast_info.get("lastPrice") or ticker.fast_info.get("regularMarketPrice")
            prev_close = ticker.fast_info.get("previousClose") or ticker.fast_info.get("regularMarketPreviousClose")

            # Fallback to intraday snapshot if fast_info is unavailable
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
        logging.warning(f"Could not fetch economic calendar: {e}")
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

    # Compute current active market session
    utc_hour = time.gmtime().tm_hour
    if 22 <= utc_hour or utc_hour < 7:
        session_state = "SESSION: TOKYO / SYDNEY ACTIVE | EUROPE CLOSED"
    elif 7 <= utc_hour < 12:
        session_state = "SESSION: LONDON ACTIVE | ASIA CLOSED"
    elif 12 <= utc_hour < 16:
        session_state = "SESSION: LONDON / NY OVERLAP (PEAK LIQUIDITY)"
    elif 16 <= utc_hour < 21:
        session_state = "SESSION: NEW YORK ACTIVE | LONDON CLOSED"
    else:
        session_state = "SESSION: OFF-PEAK HOURS / MARKET TRANSITION"

    ax.text(0.05, 0.75, "APEX INSTITUTIONAL MACRO DESK", fontsize=11, color='#8b949e', fontweight='bold', fontfamily='monospace')
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
# Formatting, Logging & Telegram Helpers
# ---------------------------------------------------------------------------
def sanitize_telegram_markdown(text: str) -> str:
    # Auto-close unclosed bold or italic tags to prevent Telegram API 400 rejection
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
    
    # Extract setup titles matching Setup #1, Setup #2 format
    setup_matches = re.findall(r"\*Setup #\d+:.*?\*", report_text)
    
    if not setup_matches:
        logging.info("No formatted trade setups detected to write into CSV log.")
        return

    try:
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Session Mode", "Setup Details"])
            
            for setup in setup_matches:
                clean_title = setup.replace("*", "").strip()
                writer.writerow([timestamp, mode, clean_title])
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
            data={'chat_id': TELEGRAM_CHANNEL_ID, 'caption': "📊 *Apex Institutional Intelligence Briefing*", 'parse_mode': 'Markdown'}, 
            files=files, 
            timeout=15
        )
        if photo_res.status_code != 200:
            logging.warning(f"Banner upload alert ({photo_res.status_code}): {photo_res.text}")
    except Exception as e:
        logging.warning(f"Banner photo dispatch exception: {e}")

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
    if not TELEGRAM_ADMIN_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_ADMIN_CHAT_ID, "text": f"⚠️ *SYSTEM ALERT*\n{message}", "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=5)
    except Exception as e: logging.error(f"Admin alert dispatch failed: {e}")

def count_setups(report_text: str) -> int:
    matches = re.findall(r"(?i)(?:setup|rank|opportunity)\s*#?\d+", report_text)
    if matches:
        normalized = [m.lower().replace(" ", "") for m in matches]
        return len(set(normalized))
    bullet_matches = re.findall(r"\n\s*(?:\d+\.|\*|\-)\s+\*Setup.*?(?:BUY|SELL|LONG|SHORT)", report_text, re.IGNORECASE)
    return len(bullet_matches)

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

                if setup_count >= 10:
                    return report_text

                logging.warning(f"Setup count validation failed ({setup_count}/10 required). Retrying prompt...")
                prompt += "\n\nCRITICAL REQUIREMENT MISSING: You MUST generate AT LEAST 10 ranked, high-conviction trade setups clearly labeled (*Setup #1*, *Setup #2*, ..., *Setup #10*) using the exact live prices provided and following the condensed setup breakdown format."

            except Exception as e:
                error_str = str(e)
                logging.error(f"API Error with model '{model_name}': {error_str}")
                
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

        if report_text and setup_count >= 10:
            return report_text

    logging.warning("All fallback models exhausted. Dispatching best available output.")
    send_admin_alert(f"Mode `{mode}` encountered Google server capacity issues across all fallback options.")
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
        error_msg = f"Pipeline execution failed for mode `{mode}`:\n`{str(e)}`"
        logging.error(error_msg, exc_info=True)
        send_admin_alert(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()