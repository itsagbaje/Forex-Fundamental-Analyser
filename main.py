import os
import sys
import argparse
import io
import re
import json
import requests
import datetime
from zoneinfo import ZoneInfo
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from google import genai

# Dynamic Timezone (Handles GMT / BST / WAT daylight saving shifts automatically)
LOCAL_TZ = ZoneInfo("Europe/London")
STATE_FILE = "setups_today.json"

# ==========================================
# 1. WATCHLIST (EXACT 14 TARGET ASSETS)
# ==========================================
WATCHLIST_TICKERS = {
    # 🥇 Major Forex Pairs & Dollar Index
    "DXY": "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    # 🪙 Commodities
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    # 📈 Major Equity Indices
    "SP500": "ES=F",
    "NAS100": "NQ=F",
    "GER40": "^GDAXI",
    # 🪙 Crypto
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD"
}

INTERMARKET_TICKERS = {
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "VIX": "^VIX"
}

# ==========================================
# 2. STATE PERSISTENCE ENGINE
# ==========================================
def load_daily_state():
    """Loads recorded trade setups from earlier sessions on the same date."""
    if not os.path.exists(STATE_FILE):
        return {}
    
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
            
        today_str = datetime.datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
        if data.get("date") == today_str:
            return data.get("sessions", {})
        else:
            print("🧹 State file is from a previous date. Resetting daily state...")
            return {}
    except Exception as e:
        print(f"⚠️ Error reading state file: {e}")
        return {}

def save_daily_state(session_mode: str, report_text: str):
    """Saves session output to disk so EOD wrap can evaluate predictions."""
    if session_mode not in ["london", "ny"]:
        return

    today_str = datetime.datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
    current_state = load_daily_state()
    
    # Extract trade setups section from report text
    setups_match = re.search(r"5\. 🎯 HIGH CONVICTION SETUPS.*", report_text, re.DOTALL)
    extracted_setups = setups_match.group(0) if setups_match else report_text[-1000:]

    current_state[session_mode] = {
        "timestamp": datetime.datetime.now(LOCAL_TZ).strftime("%H:%M %Z"),
        "setups": extracted_setups
    }

    payload = {
        "date": today_str,
        "sessions": current_state
    }

    try:
        with open(STATE_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"💾 Successfully persisted [{session_mode.upper()}] session state to {STATE_FILE}")
    except Exception as e:
        print(f"⚠️ Failed to save state file: {e}")

# ==========================================
# 3. ECONOMIC CALENDAR FETCH ENGINE
# ==========================================
def fetch_economic_calendar():
    """Fetches high-impact economic events scheduled for today."""
    try:
        url = "https://nfp.ourfxbook.com/calendar.json" # Public high-impact calendar feed
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            events = res.json()
            today_str = datetime.datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
            high_impact = [
                f"• {e.get('time', 'TBD')} | {e.get('country', '')} | {e.get('title', '')} (Impact: HIGH)"
                for e in events if e.get('date', '').startswith(today_str) and e.get('impact') == 'High'
            ]
            if high_impact:
                return "\n".join(high_impact[:5])
    except Exception:
        pass
    return "• Standard Scheduled Macro Releases / Speech Calendar Active."

# ==========================================
# 4. MARKET DATA INGESTION & FALLBACKS
# ==========================================
def classify_market_regime():
    try:
        vix_data = yf.Ticker("^VIX").history(period="5d")['Close']
        us10y_data = yf.Ticker("^TNX").history(period="5d")['Close']
        
        cur_vix = vix_data.iloc[-1] if not vix_data.empty else 18.0
        prev_vix = vix_data.iloc[0] if len(vix_data) >= 5 else cur_vix
        vix_5d_pct = ((cur_vix - prev_vix) / prev_vix) * 100 if prev_vix != 0 else 0.0
        
        cur_10y = us10y_data.iloc[-1] if not us10y_data.empty else 4.0
        prev_10y = us10y_data.iloc[-2] if len(us10y_data) >= 2 else cur_10y
        us10y_daily_move = abs(cur_10y - prev_10y)

        if cur_vix > 22.0 or vix_5d_pct > 15.0:
            regime = "RISK_DRIVEN"
            desc = f"Fear & Volatility High (VIX: {cur_vix:.2f}). Safe-haven assets like Gold and USD dominate."
        elif us10y_daily_move > 0.05:
            regime = "RATE_DRIVEN"
            desc = f"Yield Shifts Active (US 10-Year Bond Yield at {cur_10y:.2f}%). Interest rate gaps drive money flows."
        else:
            regime = "GROWTH_DIVERGENCE"
            desc = f"Calm Market (VIX: {cur_vix:.2f}). Economic health and performance gaps guide asset direction."

        return regime, desc, cur_vix, cur_10y
    except Exception as e:
        return "RATE_DRIVEN", f"Standard market environment (Data backup: {str(e)})", 18.0, 4.25

def fetch_market_data():
    price_lines = []
    momentum_triggers = []
    all_tickers = list(WATCHLIST_TICKERS.values())
    
    try:
        data = yf.download(tickers=all_tickers, period="5d", interval="15m", progress=False)['Close']
        
        for name, ticker in WATCHLIST_TICKERS.items():
            try:
                if ticker in data and not data[ticker].dropna().empty:
                    clean_series = data[ticker].dropna()
                else:
                    # Single Ticker Fallback
                    clean_series = yf.Ticker(ticker).history(period="5d", interval="15m")['Close'].dropna()

                latest_price = clean_series.iloc[-1]
                
                # Multi-Timeframe Lookbacks
                p_15m_ago = clean_series.iloc[-2] if len(clean_series) >= 2 else latest_price
                p_30m_ago = clean_series.iloc[-3] if len(clean_series) >= 3 else latest_price
                p_1h_ago  = clean_series.iloc[-5] if len(clean_series) >= 5 else latest_price
                p_24h_ago = clean_series.iloc[-96] if len(clean_series) >= 96 else clean_series.iloc[0]
                
                pct_15m = ((latest_price - p_15m_ago) / p_15m_ago) * 100 if p_15m_ago != 0 else 0.0
                pct_30m = ((latest_price - p_30m_ago) / p_30m_ago) * 100 if p_30m_ago != 0 else 0.0
                pct_1h  = ((latest_price - p_1h_ago)  / p_1h_ago)  * 100 if p_1h_ago != 0 else 0.0
                pct_24h = ((latest_price - p_24h_ago) / p_24h_ago) * 100 if p_24h_ago != 0 else 0.0
                
                # Dynamic Volatility Triggers
                if abs(pct_15m) >= 0.25:
                    dir_str = "SPIKING" if pct_15m > 0 else "DROPPING"
                    momentum_triggers.append(f"{name} ({pct_15m:+.2f}% in 15m — {dir_str})")
                elif abs(pct_30m) >= 0.35:
                    dir_str = "RALLYING" if pct_30m > 0 else "SLIDING"
                    momentum_triggers.append(f"{name} ({pct_30m:+.2f}% in 30m — {dir_str})")
                elif abs(pct_1h) >= 0.45:
                    dir_str = "SURGING" if pct_1h > 0 else "PLUMMETING"
                    momentum_triggers.append(f"{name} ({pct_1h:+.2f}% in 1h — {dir_str})")

                momentum_tag = "🔥 HIGH VOLATILITY" if (abs(pct_15m) >= 0.20 or abs(pct_30m) >= 0.30) else ("🚀 PUMPING" if pct_1h > 0.3 else ("🔻 DUMPING" if pct_1h < -0.3 else "⚖️ STABLE"))
                
                price_lines.append(
                    f"• {name}: Price = {latest_price:.4f} | 15m: {pct_15m:+.2f}% | 30m: {pct_30m:+.2f}% | 1h: {pct_1h:+.2f}% | 24h: {pct_24h:+.2f}% [{momentum_tag}]"
                )
            except Exception as item_err:
                price_lines.append(f"• {name}: Price data updating ({str(item_err)})...")
    except Exception as e:
        price_lines.append(f"Price fetch issue: {str(e)}")
        
    return "\n".join(price_lines), momentum_triggers

def fetch_intermarket_data():
    lines = []
    try:
        data = yf.download(tickers=list(INTERMARKET_TICKERS.values()), period="5d", interval="1d", progress=False)['Close']
        for name, ticker in INTERMARKET_TICKERS.items():
            if ticker in data and not data[ticker].dropna().empty:
                val = data[ticker].dropna().iloc[-1]
                lines.append(f"{name}: {val:.2f}")
    except Exception as e:
        lines.append(f"Intermarket fetch issue: {str(e)}")
    return " | ".join(lines) if lines else "Key indicators updating..."

# ==========================================
# 5. TECHNICAL CHART & BANNER GENERATOR
# ==========================================
def generate_banner(mode: str, regime: str) -> io.BytesIO:
    """Generates a technical intraday chart banner using Gold (GC=F) price action."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.2), gridspec_kw={'height_ratios': [2.2, 1]}, facecolor='#0D1117')
    
    # ------------------ TOP PANEL: INTRADAY CHART ------------------
    ax1.set_facecolor('#161B22')
    try:
        chart_df = yf.Ticker("GC=F").history(period="1d", interval="15m")
        if not chart_df.empty:
            ax1.plot(chart_df.index, chart_df['Close'], color='#58A6FF', linewidth=1.8, label="GOLD Intraday (15m)")
            ax1.fill_between(chart_df.index, chart_df['Close'], chart_df['Close'].min(), color='#58A6FF', alpha=0.08)
            ax1.set_title("XAUUSD / GOLD — INTRADAY PRICE ACTION", color='#8B949E', fontsize=9, loc='left', pad=6)
            ax1.grid(True, color='#21262D', linestyle='--', linewidth=0.5)
            ax1.tick_params(colors='#8B949E', labelsize=7)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=LOCAL_TZ))
        else:
            ax1.text(0.5, 0.5, "Live Technical Chart Active", color='#8B949E', ha='center', va='center')
    except Exception:
        ax1.text(0.5, 0.5, "Live Technical Chart Active", color='#8B949E', ha='center', va='center')
        
    # ------------------ BOTTOM PANEL: INSTITUTIONAL BADGE ------------------
    ax2.set_facecolor('#0D1117')
    ax2.axis('off')

    title = "APEX MACRO DESK"
    subtitle = "EOD WRAP & SCOREBOARD" if mode == "eod" else ("FLASH POST-NEWS UPDATE" if mode in ["update", "auto_check"] else f"{mode.upper()} SESSION INTELLIGENCE")
    regime_badge = f"ACTIVE REGIME: {regime}"
    
    now_local = datetime.datetime.now(LOCAL_TZ)
    tz_name = now_local.strftime("%Z")
    date_str = f"{now_local.strftime('%Y-%m-%d | %H:%M')} {tz_name}"

    ax2.text(0.5, 0.75, title, color='#58A6FF', fontsize=18, fontweight='bold', ha='center', va='center')
    ax2.text(0.5, 0.50, subtitle, color='#FF7B72' if mode in ["update", "auto_check"] else '#FFFFFF', fontsize=11, fontweight='bold', ha='center', va='center')
    ax2.text(0.5, 0.28, regime_badge, color='#3FB950', fontsize=9, fontweight='bold', ha='center', va='center')
    ax2.text(0.5, 0.10, date_str, color='#8B949E', fontsize=8, ha='center', va='center')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf

# ==========================================
# 6. AI MACRO ENGINE (GEMINI 3.5 FLASH)
# ==========================================
def generate_report(mode: str, intermarket_data: str, ohlc_prices: str, regime: str, regime_desc: str, calendar_events: str, past_state: dict, api_key: str, trigger_reason: str = "") -> str:
    client = genai.Client(api_key=api_key)

    now_local = datetime.datetime.now(LOCAL_TZ)
    date_str = now_local.strftime("%A, %d %b %Y").upper()
    tz_name = now_local.strftime("%Z")
    time_str = f"{now_local.strftime('%H:%M')} {tz_name}"
    date_header = f"📅 <b>DATE:</b> <code>{date_str}</code> | <code>{time_str}</code>"

    is_flash_update = mode in ["update", "auto_check"]
    
    # Format persisted setups context for EOD wrap
    history_context = ""
    if mode == "eod" and past_state:
        history_context = "\n<b>PREVIOUS SESSION SETUPS RECORDED TODAY:</b>\n"
        for s_name, s_data in past_state.items():
            history_context += f"\n--- {s_name.upper()} SESSION ({s_data.get('timestamp')}) ---\n{s_data.get('setups')}\n"
    elif mode == "eod":
        history_context = "\n(No earlier session setups recorded today in local state.)\n"

    # ----------------------------------------------------
    # A. FLASH POST-NEWS UPDATE TEMPLATE
    # ----------------------------------------------------
    if is_flash_update:
        flash_context = f"\n⚠️ <b>FLASH TRIGGER DETECTED:</b> {trigger_reason}\n" if trigger_reason else ""
        
        prompt = f"""
You are the Chief Market Analyst at Apex Macro Desk. Synthesize a **FLASH POST-NEWS INTRADAY MARKET UPDATE**.

WRITING RULES:
1. **FOCUS ON SHORT-TERM MOVES (15m, 30m, 1h):** Detail short-term price action. Explain if this is an institutional breakout, liquidity sweep, or news whipsaw.
2. **HTML STYLING ONLY:** Use <b>BOLD</b> for headers/tickers, <i>italics</i> for context, and <code>code tags</code> for key numbers/prices.
3. **DOUBLE SPACING:** Leave DOUBLE LINE BREAKS (`\n\n`) between sections and asset breakdowns.

INPUT DATA:
- **ACTIVE REGIME:** {regime} ({regime_desc})
{flash_context}
- **ECONOMIC CALENDAR TODAY:**
{calendar_events}
- **INTRADAY LOOKBACKS (15m, 30m, 1h, 24h):**
{ohlc_prices}
- **INTERMARKET INDICATORS:**
{intermarket_data}

REQUIRED TELEGRAM OUTPUT STRUCTURE:

⚡ <b>APEX MACRO DESK — FLASH POST-NEWS UPDATE</b>
{date_header}
━━━━━━━━━━━━━━━━━━━━━━

<b>1. 🔥 INTRADAY MOMENTUM & CATALYST (15m–1h WINDOW)</b>

• <b>What Just Happened:</b> <i>Provide a 3-sentence summary of price action over the last 15 to 60 minutes.</i>

• <b>Price Action Verdict:</b> <code>[INSTITUTIONAL BREAKOUT / LIQUIDITY SWEEP / NEWS-DRIVEN WHIPSAW / RETEST IN PROGRESS]</code>

• <b>Active Market Regime:</b> <i>{regime} — {regime_desc}</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>2. 🎯 TOP MOVERS & SHORT-TERM BREAKDOWNS</b>

• <b>[TICKER]</b> — <code>15m: [X%] | 30m: [Y%] | 1h: [Z%]</code>
  - <b>Short-Term Reaction:</b> <i>Explain key levels tested or broken in the last 15–60 minutes.</i>

• <b>[TICKER]</b> — <code>15m: [X%] | 30m: [Y%] | 1h: [Z%]</code>
  - <b>Short-Term Reaction:</b> <i>Explain key levels tested or broken in the last 15–60 minutes.</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>3. 🧭 IMMEDIATE TACTICAL GUIDANCE (NEXT 1–2 HOURS)</b>

• <b>USD & Yield Outlook:</b> <i>1 sentence on expected USD direction.</i>

• <b>Trade Execution Caution:</b> <i>2 sentences advising whether to chase momentum or wait for retests.</i>
"""

    # ----------------------------------------------------
    # B. END OF DAY (EOD) WRAP TEMPLATE (STATE-AWARE)
    # ----------------------------------------------------
    elif mode == "eod":
        prompt = f"""
You are the Chief Market Analyst at Apex Macro Desk. Synthesize an **END OF DAY (EOD) MACRO SCOREBOARD & NEXT-DAY OUTLOOK**.

CRITICAL EVALUATION INSTRUCTIONS:
Evaluate the REAL trade setups issued during today's London/NY sessions (provided below) against today's closing prices. Determine whether targets or stop losses were hit.

INPUT DATA:
- **ACTIVE REGIME:** {regime} ({regime_desc})
- **TODAY'S SCHEDULED NEWS EVENTS:**
{calendar_events}
- **RECORDED SESSION SETUPS FROM TODAY:**
{history_context}
- **CLOSING & INTRADAY PRICES (14 TARGET ASSETS):**
{ohlc_prices}
- **INTERMARKET INDICATORS:**
{intermarket_data}

REQUIRED TELEGRAM OUTPUT STRUCTURE:

Gl <b>APEX MACRO DESK — END OF DAY (EOD) WRAP & SCOREBOARD</b>
{date_header}
━━━━━━━━━━━━━━━━━━━━━━

<b>1. 📊 DAILY MACRO RECAP & DAY IN REVIEW</b>

• <b>Session Summary:</b> <i>3-4 sentence high-level overview of price action across London and NY.</i>

• <b>Active Market Regime:</b> <i>{regime} — {regime_desc}</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>2. ⚖️ TODAY'S SETUPS SCOREBOARD & POST-MORTEM</b>
(Evaluate today's London & NY trade setups listed in historical context above against actual closing prices)

• <b>[TICKER] — [BUY / SELL]</b>
  - <b>Outcome Verdict:</b> <code>[🎯 TARGET HIT / 🛑 STOP LOSS HIT / ⚠️ INVALIDATED / ⏳ IN PROGRESS]</code>
  - <b>Executed Zone:</b> <code>[Price Level]</code> | <b>Peak Move:</b> <code>[Price / Pips]</code>
  - <b>Post-Mortem Analysis:</b> <i>Explain in 2 sentences WHY the trade succeeded or failed based on order flow or news releases today.</i>

• <b>[TICKER] — [BUY / SELL]</b>
  - <b>Outcome Verdict:</b> <code>[🎯 TARGET HIT / 🛑 STOP LOSS HIT / ⚠️ INVALIDATED / ⏳ IN PROGRESS]</code>
  - <b>Executed Zone:</b> <code>[Price Level]</code> | <b>Peak Move:</b> <code>[Price / Pips]</code>
  - <b>Post-Mortem Analysis:</b> <i>Explain in 2 sentences WHY the trade succeeded or failed.</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>3. 🔄 INTRADAY DEVELOPMENTS & NARRATIVE SHIFT</b>

• <b>Key Catalyst Events:</b> <i>Detail 2 specific news events from today and how price reacted.</i>

• <b>Impact on Next-Day Thesis:</b> <i>Explain whether today's developments INVALIDATED, REINFORCED, or SHIFTED our higher-timeframe bias for tomorrow.</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>4. 🧭 NEXT DAY BIAS & OVERNIGHT WATCHLIST</b>

• <b>DXY & Dollar Direction:</b> 🟢 BULLISH / 🔴 BEARISH / ⚪ NEUTRAL
  ↳ <i>1 sentence explaining tomorrow's USD bias.</i>

• <b>Risk Sentiment:</b> <i>1 sentence on expected equity & crypto momentum overnight.</i>

• <b>Tomorrow's Key Pivot Levels:</b>
  - <code>GOLD</code>: Watch <code>[Key Level]</code>
  - <code>EURUSD</code>: Watch <code>[Key Level]</code>
  - <code>SP500 / BTC</code>: Watch <code>[Key Level]</code>

━━━━━━━━━━━━━━━━━━━━━━

<b>5. 💡 INSTITUTIONAL EDGE — TRADING LESSON OF THE DAY</b>

• <b>Daily Takeaway:</b> <i>Write a 2-sentence trading takeaway based on today's price behavior.</i>
"""

    # ----------------------------------------------------
    # C. STANDARD SESSION TEMPLATE (LONDON & NEW YORK)
    # ----------------------------------------------------
    else:
        prompt = f"""
You are the Chief Market Analyst at Apex Macro Desk. Create a beginner-friendly **{mode.upper()} SESSION MARKET REPORT**.

WRITING RULES:
1. **HTML STYLING ONLY:** Use <b>BOLD</b> for headers/tickers, <i>italics</i> for explanations, and <code>code tags</code> for key numbers/prices.
2. **DOUBLE SPACING:** Leave DOUBLE LINE BREAKS (`\n\n`) between sections and asset breakdowns.

INPUT DATA:
- **ACTIVE REGIME:** {regime} ({regime_desc})
- **TODAY'S ECONOMIC CALENDAR:**
{calendar_events}
- **LIVE INTRADAY PRICES & SHORT-TERM LOOKBACKS (15m, 30m, 1h, 24h):**
{ohlc_prices}
- **INTERMARKET INDICATORS:**
{intermarket_data}

REQUIRED TELEGRAM OUTPUT STRUCTURE:

🏛️ <b>APEX MACRO DESK — {mode.upper()} SESSION INTELLIGENCE</b>
{date_header}
━━━━━━━━━━━━━━━━━━━━━━

<b>1. 🌐 DECISION ENGINE & MACRO FRAMEWORK OVERVIEW</b>

• <b>Active Market Regime:</b> <i>{regime} — {regime_desc}</i>

• <b>Scheduled Economic Risk:</b> <i>2 sentences analyzing today's economic releases.</i>

• <b>Interest Rate & Bond Yield Differentials:</b> <i>2 plain English sentences on bond yield impacts.</i>

• <b>Central Bank Policy Stance:</b> <i>2 plain English sentences on Fed/ECB/BOE expectations.</i>

• <b>Market Sentiment & Volatility:</b> <i>2 plain English sentences on VIX and risk sentiment.</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>2. 🥇 MAJOR FOREX PAIRS & DXY</b>
(Evaluate ONLY: DXY, EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF)

Format each asset:

<b>EURUSD</b> 🟢 BULLISH | Price: <code>1.0850</code> (<code>1h: +0.12%</code>)
↳ <b>Macro Drivers:</b> <i>1-2 sentences explaining drivers based on yields and news.</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>3. 📈 INDICES & COMMODITIES</b>
(Evaluate ONLY: GOLD, SILVER, SP500, NAS100, GER40)

Format each asset:

<b>GOLD</b> 🟢 BULLISH | Price: <code>2400.50</code> (<code>1h: +0.35%</code>)
↳ <b>Macro Drivers:</b> <i>1-2 sentences explaining market drivers.</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>4. 🪙 CRYPTO ASSETS</b>
(Evaluate ONLY: BTCUSDT, ETHUSDT)

Format each asset:

<b>BTCUSDT</b> ⚪ NEUTRAL | Price: <code>64200.00</code> (<code>1h: -0.05%</code>)
↳ <b>Macro Drivers:</b> <i>1-2 sentences explaining market drivers.</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>5. 🎯 HIGH CONVICTION SETUPS</b>
(Pick the 2-3 best trade setups where rates, sentiment, and flows align)

• <b>[ASSET TICKER] — [BUY / SELL]</b>
  - <b>Trigger Zone:</b> <code>[Price]</code>
  - <b>Target Price:</b> <code>[Price]</code> | <b>Stop Loss:</b> <code>[Price]</code>
  - <b>Trade Game Plan:</b> <i>1 sentence explaining why rates, sentiment, and flows align.</i>
"""

    model_candidates = [
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ]
    last_exception = None

    for model_name in model_candidates:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_exception = e
            continue

    raise last_exception

# ==========================================
# 7. TELEGRAM DISPATCH ENGINE
# ==========================================
def send_telegram(bot_token: str, chat_id: str, message: str, image_buf: io.BytesIO):
    photo_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    msg_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    files = {'photo': ('banner.png', image_buf, 'image/png')}
    try:
        photo_res = requests.post(photo_url, data={'chat_id': chat_id}, files=files, timeout=15)
        if photo_res.status_code != 200:
            print(f"⚠️ Banner upload alert ({photo_res.status_code}): {photo_res.text}")
    except Exception as e:
        print(f"⚠️ Photo dispatch exception: {e}")

    MAX_LEN = 3800
    chunks = [message[i:i+MAX_LEN] for i in range(0, len(message), MAX_LEN)]

    overall_success = True
    for idx, chunk in enumerate(chunks):
        payload = {
            'chat_id': chat_id,
            'text': chunk,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        res = requests.post(msg_url, json=payload, timeout=15)

        if res.status_code != 200:
            print(f"⚠️ HTML formatting issue ({res.status_code}). Retrying chunk with plain text...")
            plain_chunk = re.sub(r'<[^>]*>', '', chunk)
            payload_plain = {
                'chat_id': chat_id,
                'text': plain_chunk,
                'disable_web_page_preview': True
            }
            res = requests.post(msg_url, json=payload_plain, timeout=15)

        if res.status_code == 200:
            print(f"   ↳ Successfully sent report part {idx + 1}/{len(chunks)}")
        else:
            print(f"❌ Telegram API Error ({res.status_code}): {res.text}")
            overall_success = False

    return overall_success

# ==========================================
# 8. CLI EXECUTION ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Apex Macro Desk Production Engine")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "london", "ny", "eod", "update", "auto_check"], help="Session mode")
    args = parser.parse_args()

    gemini_key = os.getenv("GEMINI_API_KEY")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not gemini_key or not bot_token or not chat_id:
        print("ERROR: GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, and TELEGRAM_CHAT_ID must be set.")
        sys.exit(1)

    print("🧠 Running Regime Classifier...")
    regime, regime_desc, vix_val, yield_val = classify_market_regime()
    print(f"   ↳ Active Regime: {regime} | VIX: {vix_val:.2f} | US10Y: {yield_val:.2f}%")

    print("📅 Fetching Scheduled High-Impact Economic News Events...")
    calendar_events = fetch_economic_calendar()

    print("📊 Ingesting Market Data (14 Assets — 15m, 30m, 1h Lookbacks)...")
    ohlc_prices, momentum_triggers = fetch_market_data()

    print("🌐 Ingesting Intermarket Data & Yields...")
    intermarket_data = fetch_intermarket_data()

    print("💾 Loading Historical Daily State...")
    past_state = load_daily_state()

    # Automatic Volatility Evaluation
    trigger_reason = ""
    if args.mode == "auto_check":
        if momentum_triggers or vix_val > 22.0:
            reasons = momentum_triggers + ([f"VIX Spike ({vix_val:.2f})"] if vix_val > 22.0 else [])
            trigger_reason = ", ".join(reasons)
            print(f"⚡ VOLATILITY TRIGGER DETECTED: {trigger_reason}")
            modes_to_run = ["update"]
        else:
            print("⚖️ Market stable. No significant short-term move detected. Skipping automatic broadcast.")
            sys.exit(0)
    elif args.mode == "all":
        modes_to_run = ["london", "ny", "eod"]
    else:
        modes_to_run = [args.mode]

    for current_mode in modes_to_run:
        print(f"\n🚀 Processing [{current_mode.upper()} SESSION]...")

        print("🤖 Synthesizing Institutional Macro Report with Gemini 3.5 Flash...")
        report_text = generate_report(current_mode, intermarket_data, ohlc_prices, regime, regime_desc, calendar_events, past_state, gemini_key, trigger_reason)

        print("🖼️ Generating Technical Chart & Session Banner Graphic...")
        banner_buf = generate_banner(current_mode, regime)

        print("📡 Broadcasting Session Intelligence directly to Telegram...")
        success = send_telegram(bot_token, chat_id, report_text, banner_buf)

        if success:
            print(f"✅ [{current_mode.upper()}] Report broadcasted successfully!")
            # Persist trade setup predictions for EOD evaluation
            save_daily_state(current_mode, report_text)
        else:
            print(f"❌ [{current_mode.upper()}] Broadcast failed.")
            sys.exit(1)

if __name__ == "__main__":
    main()
