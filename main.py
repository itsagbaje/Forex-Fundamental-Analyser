import argparse
import datetime
import json
import os
import sys
import io
import requests
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from google import genai
from google.genai import errors

# ==============================================================================
# CONFIGURATION & ASSET WATCHLIST
# ==============================================================================
ASSETS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "NZDUSD": "NZDUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "SP500": "^GSPC",
    "NAS100": "^NDX",
    "DXY": "DX-Y.NYB",
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "GER40": "^GDAXI"
}

# Updated model candidate queue prioritizing latest releases
MODEL_CANDIDATES = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro",
    "gemini-2.5-flash"
]

STATE_FILE = "daily_state.json"

# ==============================================================================
# 10-COMPONENT FRAMEWORK: REGIME, YIELDS & MACRO DATA
# ==============================================================================
def get_regime_and_yields():
    """Classifies the market macro regime using VIX and US10Y Treasury yields."""
    print("🧠 Running Regime Classifier...")
    try:
        vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]
        us10y = yf.Ticker("^TNX").history(period="1d")["Close"].iloc[-1]
    except Exception:
        vix, us10y = 15.0, 4.5  # Fallback defaults

    if vix > 22.0:
        regime = "RISK_OFF / LIQUIDITY_SQUEEZE"
        desc = "Elevated volatility driving safe-haven demand."
    elif us10y > 4.5 and vix < 18.0:
        regime = "GROWTH_DIVERGENCE"
        desc = "US yield advantage underpinning dollar strength while risk assets chop."
    else:
        regime = "RISK_ON / CARRY_SEEKING"
        desc = "Stable macroeconomic backdrop favoring high-beta assets."

    print(f"   ↳ Active Regime: {regime} | VIX: {vix:.2f} | US10Y: {us10y:.2f}%")
    return regime, desc, {"VIX": round(vix, 2), "US10Y": round(us10y, 2)}


def fetch_economic_calendar():
    """Fetches high-impact fundamental drivers for the current trading window."""
    print("📅 Fetching Scheduled High-Impact Economic News Events...")
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    events = []
    
    if finnhub_key:
        try:
            today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
            url = f"https://finnhub.io/api/v1/calendar/economic?from={today}&to={today}&token={finnhub_key}"
            resp = requests.get(url, timeout=10)
            if resp.ok:
                data = resp.json()
                for item in data.get("economicCalendar", []):
                    if item.get("impact", "").lower() in ["high", "3"]:
                        events.append({
                            "time": item.get("time"),
                            "currency": item.get("country"),
                            "event": item.get("event"),
                            "impact": "HIGH"
                        })
        except Exception as e:
            print(f"⚠️ Finnhub calendar fetch failed: {e}")

    if not events:
        # Fallback placeholder if no API key or events found
        events.append({"time": "All Day", "currency": "GLOBAL", "event": "No Tier-1 Scheduled Red-Folder Prints", "impact": "MEDIUM"})
        
    return events


def ingest_market_data():
    """Ingests multi-timeframe lookbacks across the 14-asset institutional watchlist."""
    print("📊 Ingesting Market Data (14 Assets — 15m, 30m, 1h Lookbacks)...")
    data = {}
    for symbol, ticker in ASSETS.items():
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="3d", interval="15m")
            if not df.empty:
                last_price = df["Close"].iloc[-1]
                prev_price = df["Close"].iloc[-2]
                change_pct = ((last_price - prev_price) / prev_price) * 100
                data[symbol] = {
                    "price": round(last_price, 5 if "USD" in symbol else 2),
                    "change_pct": round(change_pct, 2),
                    "high_1h": round(df["High"].iloc[-4:].max(), 5 if "USD" in symbol else 2),
                    "low_1h": round(df["Low"].iloc[-4:].min(), 5 if "USD" in symbol else 2)
                }
        except Exception as e:
            data[symbol] = {"price": "N/A", "change_pct": 0.0, "error": str(e)}
    return data


def load_historical_state():
    """Loads historical state file to maintain session continuity."""
    print("💾 Loading Historical Daily State...")
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_run": str(datetime.datetime.utcnow()), "notes": "Cold start state initialization."}


def save_historical_state(state_data):
    """Persists current state updates for consecutive session runs."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state_data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save state: {e}")

# ==============================================================================
# BANNER GRAPHIC GENERATION
# ==============================================================================
def generate_session_banner(session_name, regime_text):
    """Generates an institutional dark-mode visual banner for Telegram reports."""
    print("🎨 Generating Dark-Mode Session Banner Graphic...")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    # Styling accents
    ax.text(0.05, 0.70, f"APEX MACRO INTELLIGENCE", fontsize=12, color='#8b949e', fontweight='bold', fontfamily='monospace')
    ax.text(0.05, 0.40, f"{session_name.upper()} SESSION BRIEF", fontsize=24, color='#ffffff', fontweight='bold', fontfamily='sans-serif')
    ax.text(0.05, 0.15, f"REGIME: {regime_text}", fontsize=10, color='#58a6ff', fontweight='bold', fontfamily='monospace')

    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

# ==============================================================================
# REPORT SYNTHESIS (GEMINI ENGINE)
# ==============================================================================
def generate_report(current_mode, intermarket_data, ohlc_prices, regime, regime_desc, calendar_events, past_state, gemini_key, trigger_reason):
    """Synthesizes institutional macro breakdown using fallback model queue."""
    print(f"\n🚀 Processing [{current_mode.upper()} SESSION]...")
    
    client = genai.Client(api_key=gemini_key)
    
    prompt = f"""
    You are an Elite Foreign Exchange & Macro Strategist publishing a high-conviction briefing for institutional desks.
    
    SESSION: {current_mode.upper()}
    TRIGGER: {trigger_reason}
    MACRO REGIME: {regime} ({regime_desc})
    INTERMARKET YIELDS: {json.dumps(intermarket_data)}
    
    MARKET WATCHLIST SNAPSHOT (14 ASSETS):
    {json.dumps(ohlc_prices, indent=2)}
    
    SCHEDULED MACRO EVENTS:
    {json.dumps(calendar_events, indent=2)}
    
    HISTORICAL STATE:
    {json.dumps(past_state, indent=2)}
    
    REQUIREMENTS:
    Produce an executive-grade market analysis breakdown structured clearly for Telegram markdown:
    1. **Executive Macro Summary & Cross-Asset Flow**
    2. **Tier-1 Event Risk Assessment**
    3. **Asset Class Deep-Dive Matrix (Forex, Commodities, Indices, Crypto)**
    4. **Session Bias & Key Execution Levels**
    """

    last_exception = None
    for model_id in MODEL_CANDIDATES:
        try:
            print(f"🤖 Synthesizing Institutional Macro Report with {model_id}...")
            response = client.models.generate_content(
                model=model_id,
                contents=prompt
            )
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"⚠️ Model {model_id} failed: {e}. Trying next fallback...")
            last_exception = e

    if last_exception:
        raise last_exception

# ==============================================================================
# TELEGRAM PUBLISHING & MAIN ENTRYWAY
# ==============================================================================
def send_telegram_media_group_or_message(text, banner_buf, bot_token, chat_id):
    """Dispatches visual banner and formatted text report to Telegram."""
    if not bot_token or not chat_id:
        print("⚠️ Telegram tokens missing. Dumping to stdout:\n")
        print(text)
        return

    # Send Banner Image first
    banner_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {'photo': ('banner.png', banner_buf.getvalue(), 'image/png')}
    data = {'chat_id': chat_id, 'caption': f"📊 *Apex Macro Briefing — {datetime.datetime.utcnow().strftime('%Y-%m-%d')}*", 'parse_mode': 'Markdown'}
    
    img_resp = requests.post(banner_url, data=data, files=files)
    if not img_resp.ok:
        print(f"⚠️ Banner dispatch failed: {img_resp.text}")

    # Send Text Report (Splitting chunks if exceeding Telegram 4096 character cap)
    msg_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    chunk_size = 4000
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        payload = {"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"}
        resp = requests.post(msg_url, json=payload)
        if not resp.ok:
            print(f"❌ Telegram chunk dispatch failed: {resp.status_code} - {resp.text}")
        else:
            print("✅ Report chunk successfully broadcasted to Telegram.")


def main():
    parser = argparse.ArgumentParser(description="Apex Macro & Forex Fundamental Analyser")
    parser.add_argument("--mode", type=str, default="london", help="Session mode: asian, london, newyork, eod")
    parser.add_argument("--trigger", type=str, default="Automated Cron Dispatch", help="Execution trigger description")
    args = parser.parse_args()

    gemini_key = os.getenv("GEMINI_API_KEY")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not gemini_key:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    regime, regime_desc, intermarket_data = get_regime_and_yields()
    calendar_events = fetch_economic_calendar()
    ohlc_prices = ingest_market_data()
    past_state = load_historical_state()

    # Generate visual dark-mode banner
    banner_buf = generate_session_banner(args.mode, regime)

    # Run AI Model Synthesis Pipeline
    report_text = generate_report(
        current_mode=args.mode,
        intermarket_data=intermarket_data,
        ohlc_prices=ohlc_prices,
        regime=regime,
        regime_desc=regime_desc,
        calendar_events=calendar_events,
        past_state=past_state,
        gemini_key=gemini_key,
        trigger_reason=args.trigger
    )

    # Save current run state update
    save_historical_state({
        "last_run": str(datetime.datetime.utcnow()),
        "mode": args.mode,
        "regime": regime,
        "status": "Success"
    })

    # Broadcast to Telegram channel
    send_telegram_media_group_or_message(report_text, banner_buf, bot_token, chat_id)


if __name__ == "__main__":
    main()
