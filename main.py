import os
import sys
import requests
from datetime import datetime
from google import genai

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FMP_API_KEY = os.getenv("FMP_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Environment Variable Guard Check
if not GEMINI_API_KEY:
    print("[X] Critical Error: GEMINI_API_KEY environment variable is missing or empty.")
    print("Please verify your GitHub Repository Secrets name matches 'GEMINI_API_KEY'.")
    sys.exit(1)

WATCHLIST = [
    "EURUSD", "GBPUSD", "USDCHF", "USDCAD", "GBPJPY", "EURJPY",
    "JP225", "NAS100", "SP500", "NZDUSD", "AUDUSD", "EURCAD",
    "GBPCAD", "GOLD", "SILVER", "GER40", "BTCUSD", "ETHUSD"
]

TARGET_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"}

# Initialize Official Gemini Client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 2. ECONOMIC CALENDAR DATA FETCHING
# ==========================================
def fetch_today_economic_events() -> str:
    """Fetches real-time economic calendar from Financial Modeling Prep API."""
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today_str}&to={today_str}&apikey={FMP_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return "Unable to pull live calendar data. Falling back to macro model estimates."
        
        events = response.json()
        if not isinstance(events, list) or not events:
            return "No major macroeconomic events scheduled for today."
            
        filtered_events = []
        for ev in events:
            currency = str(ev.get("currency", "")).upper()
            impact = str(ev.get("impact", "")).capitalize()
            event_name = ev.get("event", "Economic Event")
            
            if currency in TARGET_CURRENCIES and impact in ["High", "Medium"]:
                actual = ev.get("actual") if ev.get("actual") is not None else "Pending"
                estimate = ev.get("estimate") if ev.get("estimate") is not None else "N/A"
                previous = ev.get("previous") if ev.get("previous") is not None else "N/A"
                date_time = ev.get("date", "")
                
                filtered_events.append(
                    f"• [{currency} | {impact} Impact] {event_name} @ {date_time} UTC "
                    f"(Actual: {actual} | Forecast: {estimate} | Prev: {previous})"
                )
                
        if not filtered_events:
            return "No high/medium impact calendar events scheduled for target currencies today."
            
        return "\n".join(filtered_events[:15])
        
    except Exception as err:
        return f"Economic calendar fetch error: {err}"

# ==========================================
# 3. REPORT GENERATION VIA GEMINI
# ==========================================
def generate_report_with_calendar(session_name: str, raw_calendar_data: str) -> str:
    prompt = f"""
You are an institutional macro quantitative strategist. Generate a **{session_name} Session Fundamental Analysis Report**.

Watchlist Assets: {", ".join(WATCHLIST)}

LIVE ECONOMIC CALENDAR DATA FOR TODAY:
{raw_calendar_data}

**STRICT FORMATTING RULES:**
1. Use ONLY valid Telegram HTML tags: <b>bold</b>, <i>italic</i>, <code>code</code>, <pre>preformatted</pre>.
2. Do NOT use markdown asterisks (*). Do NOT use raw < or > unless forming valid HTML tags.

**REQUIRED REPORT SECTIONS:**
1. <b>🏛️ {session_name.upper()} SESSION MACRO & LIVE CALENDAR</b>
2. <b>📊 INSTITUTIONAL SCORECARD (-10 to +10)</b> (Include all 18 watchlist assets)
3. <b>🎯 TOP CONFLUENCE SETUPS</b>
4. <b>⚠️ CORRELATION & EXPOSURE GUARDRAIL</b>
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    return response.text

# ==========================================
# 4. TELEGRAM DISPATCH HELPER
# ==========================================
def send_telegram_report(html_content: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Chunk long reports to fit Telegram's limit
    chunks = [html_content[i:i+4000] for i in range(0, len(html_content), 4000)]

    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        res = requests.post(url, json=payload, timeout=15)
        if res.status_code != 200:
            raise RuntimeError(f"Telegram Delivery Error ({res.status_code}): {res.text}")
            
    print("[✓] Report successfully delivered to Telegram.")

# ==========================================
# 5. MAIN EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    utc_hour = datetime.utcnow().hour
    session = "London" if utc_hour < 12 else "New York"

    print(f"[*] Fetching real-time economic calendar...")
    calendar_context = fetch_today_economic_events()
    
    print(f"[*] Generating {session} Session Report using Gemini API...")
    try:
        report_html = generate_report_with_calendar(session, calendar_context)
        send_telegram_report(report_html)
    except Exception as e:
        print(f"[X] Pipeline execution failed: {e}")
        sys.exit(1)
