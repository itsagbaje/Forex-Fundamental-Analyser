import os
import sys
import io
import re
import requests
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import cot_reports as cot
from google import genai

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FMP_API_KEY = os.getenv("FMP_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("[X] Critical Error: GEMINI_API_KEY missing.")
    sys.exit(1)

WATCHLIST_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "GBPJPY": "GBPJPY=X",
    "EURJPY": "EURJPY=X",
    "JP225": "^N225",
    "NAS100": "^NDX",
    "SP500": "^GSPC",
    "NZDUSD": "NZDUSD=X",
    "AUDUSD": "AUDUSD=X",
    "EURCAD": "EURCAD=X",
    "GBPCAD": "GBPCAD=X",
    "GOLD": "XAUUSD=X",
    "SILVER": "XAGUSD=X",
    "GER40": "^GDAXI",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD"
}

TARGET_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"}
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 2. DATA FETCHING MODULES
# ==========================================
def fetch_intermarket_data() -> str:
    symbols = {"US Dollar Index (DXY)": "DX=F", "US 10Y Yield": "^TNX"}
    summary = []
    for name, sym in symbols.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                prev = hist['Close'].iloc[-2]
                curr = hist['Close'].iloc[-1]
                pct = ((curr - prev) / prev) * 100
                summary.append(f"• {name}: {curr:.2f} ({pct:+.2f}% today)")
        except Exception:
            continue
    return "\n".join(summary) if summary else "Intermarket yield data unavailable."

def fetch_watchlist_prices_and_ranges() -> str:
    """Fetches Open, High, Low, Close (OHLC) for EOD audit and intraday tracking."""
    summary = []
    print("[*] Fetching OHLC price ranges for watchlist assets...")
    
    for name, sym in WATCHLIST_TICKERS.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="1d")
            if not hist.empty:
                open_p = hist['Open'].iloc[-1]
                high_p = hist['High'].iloc[-1]
                low_p = hist['Low'].iloc[-1]
                close_p = hist['Close'].iloc[-1]
                pct = ((close_p - open_p) / open_p) * 100
                
                fmt = ".2f" if close_p > 10 else ".5f"
                summary.append(
                    f"• {name}: Open {open_p:{fmt}} | High {high_p:{fmt}} | Low {low_p:{fmt}} | Close {close_p:{fmt}} ({pct:+.2f}%)"
                )
        except Exception as e:
            print(f"[!] Warning: Could not fetch price for {name}: {e}")
            continue
            
    return "\n".join(summary) if summary else "Price data unavailable."

def fetch_today_economic_events() -> str:
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today_str}&to={today_str}&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200 or not isinstance(response.json(), list):
            return "No high impact calendar events scheduled today."
            
        filtered = []
        for ev in response.json():
            curr = str(ev.get("currency", "")).upper()
            imp = str(ev.get("impact", "")).capitalize()
            if curr in TARGET_CURRENCIES and imp in ["High", "Medium"]:
                filtered.append(
                    f"• [{curr} | {imp}] {ev.get('event')} (Actual: {ev.get('actual', 'Pending')} | Est: {ev.get('estimate', 'N/A')})"
                )
        return "\n".join(filtered[:15]) if filtered else "No major events scheduled."
    except Exception as err:
        return f"Calendar fetch error: {err}"

# ==========================================
# 3. GRAPHIC GENERATOR
# ==========================================
def generate_and_send_scorecard_image(title_suffix: str):
    fig, ax = plt.subplots(figsize=(8, 2), dpi=200)
    fig.patch.set_facecolor('#0B132B')
    ax.set_facecolor('#0B132B')
    
    ax.text(0.5, 0.65, "APEX MACRO DESK", color='#D4AF37', fontsize=18, fontweight='bold', ha='center')
    ax.text(0.5, 0.35, title_suffix.upper(), color='#00F0FF', fontsize=9, fontweight='bold', ha='center')
    ax.text(0.5, 0.15, f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", color='#FFFFFF', fontsize=7, ha='center')
    
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': ('scorecard.png', buf, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': f'🏛️ <b>APEX MACRO DESK — {title_suffix.upper()}</b>', 'parse_mode': 'HTML'}
    requests.post(url, data=data, files=files, timeout=15)

# ==========================================
# 4. REPORT GENERATION (SESSION vs EOD RECAP)
# ==========================================
def generate_report(mode: str, calendar_data: str, intermarket_data: str, ohlc_prices: str) -> str:
    if mode == "EOD":
        prompt = f"""
You are the Lead Macro Risk Officer at Apex Macro Desk. Generate the **END OF DAY SETUP PERFORMANCE & MACRO RECAP**.

TODAY'S ACTUAL INTRADAY PRICE RANGES (OHLC):
{ohlc_prices}

INTERMARKET & YIELD CLOSING SUMMARY:
{intermarket_data}

ECONOMIC CALENDAR OUTCOMES:
{calendar_data}

**STRICT FORMATTING RULES (TELEGRAM HTML):**
1. Use ONLY valid HTML tags: <b>bold</b>, <i>italic</i>, <code>code</code>, <pre>preformatted</pre>.
2. Do NOT use markdown asterisks (*). EVERY <b> tag MUST have a corresponding </b> tag.
3. Do NOT use raw '<' or '>' characters.

**REQUIRED SECTIONS:**
1. <b>🏛️ END OF DAY MACRO SUMMARY</b> (Brief overview of how economic events and yield movements drove risk sentiment today across London and NY)
2. <b>🎯 TOP CONFLUENCE SETUPS AUDIT (LONDON & NY)</b> (Evaluate the top watchlist setups generated today. State: Pair | Session | Intended Bias | Actual Range Hit | Verdict: ✅ Played Out / ❌ Invalidated / ⏸️ Consolidated)
3. <b>📊 DAY CLOSING SCORECARD REVIEW</b> (Brief summary of key gainers/losers relative to fundamental bias)
4. <b>🔮 OVERNIGHT & ASIAN SESSION OUTLOOK</b>
"""
    else:
        prompt = f"""
You are an institutional macro strategist at Apex Macro Desk. Generate the **{mode} Session Report**.

LIVE REAL-TIME WATCHLIST SPOT PRICES:
{ohlc_prices}

REAL-TIME INTERMARKET YIELDS & DXY:
{intermarket_data}

LIVE ECONOMIC CALENDAR:
{calendar_data}

**STRICT FORMATTING RULES (TELEGRAM HTML):**
1. Use ONLY valid HTML tags: <b>bold</b>, <i>italic</i>, <code>code</code>, <pre>preformatted</pre>.
2. Do NOT use markdown asterisks (*). EVERY <b> tag MUST have a corresponding </b> tag.
3. Anchor execution zones strictly to the live prices provided.

**REQUIRED SECTIONS:**
1. <b>🏛️ {mode.upper()} SESSION MACRO & YIELD DRIVERS</b>
2. <b>📊 INSTITUTIONAL SCORECARD (-10 to +10)</b>
3. <b>🔥 TOP 5 TO 7 FOCUS ASSETS (FUNDAMENTAL CONFLUENCE)</b>
4. <b>🎯 TOP CONFLUENCE SETUPS</b>
5. <b>⚠️ CORRELATION & EXPOSURE GUARDRAIL</b>
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text

# ==========================================
# 5. TELEGRAM DISPATCH & MAIN EXECUTION
# ==========================================
def send_telegram_text(html_content: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [html_content[i:i+4000] for i in range(0, len(html_content), 4000)]

    for chunk in chunks:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": True}
        res = requests.post(url, json=payload, timeout=15)
        
        if res.status_code == 400 and "can't parse entities" in res.text:
            print("[!] HTML parse error detected. Falling back to plain text...")
            plain_text = re.sub(r'<[^>]*>', '', chunk)
            payload_fallback = {"chat_id": TELEGRAM_CHAT_ID, "text": plain_text, "disable_web_page_preview": True}
            requests.post(url, json=payload_fallback, timeout=15)
        elif res.status_code != 200:
            raise RuntimeError(f"Telegram Error: {res.text}")

if __name__ == "__main__":
    current_hour = datetime.utcnow().hour
    
    # Determine mode based on UTC run time
    if current_hour >= 20:
        mode = "EOD"
        banner_title = "End of Day Setup Performance Recap"
    elif current_hour >= 12:
        mode = "New York"
        banner_title = "New York Session Intelligence"
    else:
        mode = "London"
        banner_title = "London Session Intelligence"
        
    print(f"[*] Executing Apex Macro Pipeline for {mode} Mode...")
    
    try:
        generate_and_send_scorecard_image(banner_title)
        
        market_ctx = fetch_intermarket_data()
        ohlc_ctx = fetch_watchlist_prices_and_ranges()
        cal_ctx = fetch_today_economic_events()
        
        report = generate_report(mode, cal_ctx, market_ctx, ohlc_ctx)
        
        send_telegram_text(report)
        print(f"[✓] Apex Macro {mode} Report delivered successfully.")
    except Exception as e:
        print(f"[X] Pipeline execution failed: {e}")
        sys.exit(1)
