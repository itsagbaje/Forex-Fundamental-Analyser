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

# Exact ticker mappings for all 18 watchlist assets
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
    """Fetches real-time DXY and US10Y Yields via yfinance."""
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
            
    return "\n".join(summary) if summary else "Intermarket yield data currently unavailable."

def fetch_watchlist_prices() -> str:
    """Fetches real-time prices and 24h momentum for all 18 watchlist assets."""
    summary = []
    print("[*] Fetching live quotes for all 18 watchlist assets...")
    
    for name, sym in WATCHLIST_TICKERS.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="2d")
            if len(hist) >= 1:
                curr = hist['Close'].iloc[-1]
                if len(hist) >= 2:
                    prev = hist['Close'].iloc[-2]
                    pct = ((curr - prev) / prev) * 100
                    fmt_price = f"{curr:.2f}" if curr > 10 else f"{curr:.5f}"
                    summary.append(f"• {name}: {fmt_price} ({pct:+.2f}%)")
                else:
                    fmt_price = f"{curr:.2f}" if curr > 10 else f"{curr:.5f}"
                    summary.append(f"• {name}: {fmt_price}")
        except Exception as e:
            print(f"[!] Warning: Could not fetch price for {name}: {e}")
            continue
            
    return "\n".join(summary) if summary else "Watchlist live price data unavailable."

def fetch_today_economic_events() -> str:
    """Fetches economic calendar from FMP API."""
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
                    f"• [{curr} | {imp}] {ev.get('event')} @ {ev.get('date')} UTC "
                    f"(Actual: {ev.get('actual', 'Pending')} | Est: {ev.get('estimate', 'N/A')})"
                )
        return "\n".join(filtered[:15]) if filtered else "No major events scheduled."
    except Exception as err:
        return f"Calendar fetch error: {err}"

def fetch_cot_positioning() -> str:
    """Fetches current year's CFTC Traders in Financial Futures (TFF) positioning."""
    try:
        current_year = datetime.utcnow().year
        df = cot.cot_year(year=current_year, cot_report_type='traders_in_financial_futures_fut')
        
        if df.empty:
            return "CFTC positioning data currently unavailable."

        latest_date = df['As_of_Date_In_Form_YYMMDD'].iloc[-1]
        latest_df = df[df['As_of_Date_In_Form_YYMMDD'] == latest_date]

        contracts = {
            "USD (DXY)": "CHICAGO MERCANTILE EXCHANGE",
            "EUR": "EURO FX",
            "GBP": "BRITISH POUND",
            "JPY": "JAPANESE YEN",
            "GOLD": "GOLD"
        }

        summary = [f"[CFTC Institutional Positioning - Week of {latest_date}]"]
        for label, match in contracts.items():
            row = latest_df[latest_df['Market_and_Exchange_Names'].str.contains(match, case=False, na=False)]
            if not row.empty:
                longs = row['Lev_Money_Positions_Long_All'].values[0]
                shorts = row['Lev_Money_Positions_Short_All'].values[0]
                net = longs - shorts
                sentiment = "Net Long 🟢" if net > 0 else "Net Short 🔴"
                summary.append(f"• {label}: {sentiment} ({net:+,d} contracts)")

        return "\n".join(summary) if summary else "CFTC positioning data bypassed."
    except Exception as err:
        return f"CFTC COT data fetch bypassed: {err}"

# ==========================================
# 3. SCORECARD GRAPHIC GENERATOR
# ==========================================
def generate_and_send_scorecard_image():
    """Generates a professional dark-mode graphic banner for the desk."""
    fig, ax = plt.subplots(figsize=(8, 2), dpi=200)
    fig.patch.set_facecolor('#0B132B')
    ax.set_facecolor('#0B132B')
    
    ax.text(0.5, 0.65, "APEX MACRO DESK", color='#D4AF37', fontsize=18, fontweight='bold', ha='center')
    ax.text(0.5, 0.35, "INSTITUTIONAL CROSS-ASSET INTELLIGENCE", color='#00F0FF', fontsize=9, fontweight='bold', ha='center')
    ax.text(0.5, 0.15, f"Session Briefing | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", color='#FFFFFF', fontsize=7, ha='center')
    
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {'photo': ('scorecard.png', buf, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': '🏛️ <b>APEX MACRO DESK — SESSION BRIEFING</b>', 'parse_mode': 'HTML'}
    requests.post(url, data=data, files=files, timeout=15)

# ==========================================
# 4. REPORT GENERATION VIA GEMINI
# ==========================================
def generate_report(session_name: str, calendar_data: str, intermarket_data: str, watchlist_prices: str, cot_data: str) -> str:
    prompt = f"""
You are an institutional macro strategist at Apex Macro Desk. Generate the **{session_name} Session Report**.

LIVE REAL-TIME WATCHLIST SPOT PRICES:
{watchlist_prices}

REAL-TIME INTERMARKET YIELDS & DXY:
{intermarket_data}

WEEKLY INSTITUTIONAL COT POSITIONING:
{cot_data}

LIVE ECONOMIC CALENDAR:
{calendar_data}

**STRICT FORMATTING & ACCURACY RULES:**
1. Use ONLY these valid HTML tags: <b>bold</b>, <i>italic</i>, <code>code</code>, <pre>preformatted</pre>.
2. Do NOT use markdown asterisks (*). EVERY <b> tag MUST have a corresponding </b> tag.
3. Do NOT use raw '<' or '>' characters anywhere in the text (e.g., write 'below -5' instead of '< -5').
4. **SPOT PRICE ANCHORING:** You MUST anchor all key execution levels, support/resistance references, and price comments strictly around the provided **LIVE REAL-TIME WATCHLIST SPOT PRICES**. Do NOT invent or estimate prices outside these real-time numbers.

**REQUIRED SECTIONS:**
1. <b>🏛️ {session_name.upper()} SESSION MACRO & YIELD DRIVERS</b>
2. <b>📊 INSTITUTIONAL SCORECARD (-10 to +10)</b> (List all 18 watchlist assets with signed scores and 1-word fundamental drivers)
3. <b>🔥 TOP 5 TO 7 FOCUS ASSETS (FUNDAMENTAL CONFLUENCE)</b> (Identify specifically 5 to 7 high-conviction assets; state Pair, Bias, Fundamental Justification, Key Execution Zone anchored to live price)
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
        
        # Failsafe fallback if Gemini produces malformed HTML
        if res.status_code == 400 and "can't parse entities" in res.text:
            print("[!] Telegram HTML parse error detected. Falling back to plain text dispatch...")
            plain_text = re.sub(r'<[^>]*>', '', chunk)
            payload_fallback = {"chat_id": TELEGRAM_CHAT_ID, "text": plain_text, "disable_web_page_preview": True}
            res_fb = requests.post(url, json=payload_fallback, timeout=15)
            if res_fb.status_code != 200:
                raise RuntimeError(f"Telegram Fallback Error: {res_fb.text}")
        elif res.status_code != 200:
            raise RuntimeError(f"Telegram Error: {res.text}")

if __name__ == "__main__":
    session = "London" if datetime.utcnow().hour < 12 else "New York"
    print(f"[*] Executing Apex Macro Pipeline for {session} Session...")
    
    try:
        generate_and_send_scorecard_image()
        
        market_ctx = fetch_intermarket_data()
        watchlist_ctx = fetch_watchlist_prices()
        cal_ctx = fetch_today_economic_events()
        cot_ctx = fetch_cot_positioning()
        
        report = generate_report(session, cal_ctx, market_ctx, watchlist_ctx, cot_ctx)
        
        send_telegram_text(report)
        print("[✓] Apex Macro Report delivered successfully.")
    except Exception as e:
        print(f"[X] Pipeline execution failed: {e}")
        sys.exit(1)
