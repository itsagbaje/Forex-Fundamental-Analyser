import os
import sys
import argparse
import io
import re
import requests
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from google import genai

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
# 2. DECISION ENGINE & REGIME CLASSIFIER
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
    all_tickers = list(WATCHLIST_TICKERS.values())
    try:
        data = yf.download(tickers=all_tickers, period="5d", interval="1d", progress=False)['Close']
        for name, ticker in WATCHLIST_TICKERS.items():
            if ticker in data and not data[ticker].dropna().empty:
                latest_price = data[ticker].dropna().iloc[-1]
                prev_price = data[ticker].dropna().iloc[-2] if len(data[ticker].dropna()) > 1 else latest_price
                change_pct = ((latest_price - prev_price) / prev_price) * 100
                price_lines.append(f"• {name}: Price = {latest_price:.4f} (24h Change: {change_pct:+.2f}%)")
            else:
                price_lines.append(f"• {name}: Price updating...")
    except Exception as e:
        price_lines.append(f"Price fetch issue: {str(e)}")
    return "\n".join(price_lines)

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
# 3. SESSION-ALIGNED & VOLATILITY-SORTED NEWS
# ==========================================
def fetch_economic_calendar(api_key, active_mode="all"):
    if not api_key:
        return "Economic calendar API key missing. Relying on yield trends.", False

    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today}&to={today}&apikey={api_key}"
    relevant_currencies = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD"}

    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return f"Calendar fetch failed (Status {res.status_code})", False

        events = res.json()
        filtered_events = [
            e for e in events
            if e.get("impact") in ["High", "Medium"] and e.get("currency") in relevant_currencies
        ]

        if not filtered_events:
            return "No high or medium impact economic releases scheduled for today.", False

        parsed_events = []
        has_tier1 = False

        for ev in filtered_events:
            impact_raw = ev.get("impact", "Medium")
            impact_score = 1 if impact_raw == "High" else 2
            if impact_raw == "High":
                has_tier1 = True

            date_str = ev.get("date", "")
            time_str = "TBD"
            hour = 12

            if date_str:
                try:
                    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    time_str = dt.strftime("%H:%M UTC")
                    hour = dt.hour
                except ValueError:
                    time_str = "TBD"

            if 0 <= hour < 7:
                session_name = "🌏 ASIAN SESSION (00:00 - 07:00 UTC)"
            elif 7 <= hour < 13:
                session_name = "🇬🇧 LONDON SESSION (07:00 - 13:00 UTC)"
            else:
                session_name = "🇺🇸 NEW YORK SESSION (13:00 - 21:00 UTC)"

            parsed_events.append({
                "currency": ev.get("currency"),
                "event": ev.get("event"),
                "impact": impact_raw,
                "impact_score": impact_score,
                "time_str": time_str,
                "hour": hour,
                "estimate": ev.get("estimate", "N/A"),
                "previous": ev.get("previous", "N/A"),
                "session_name": session_name
            })

        parsed_events.sort(key=lambda x: (x["hour"], x["impact_score"]))

        grouped = {}
        for ev in parsed_events:
            s_name = ev["session_name"]
            if s_name not in grouped:
                grouped[s_name] = []
            grouped[s_name].append(ev)

        output_lines = []
        for session_title, ev_list in grouped.items():
            output_lines.append(f"<b>{session_title}</b>")
            for e in ev_list:
                icon = "🔴 HIGH" if e["impact"] == "High" else "🟡 MED"
                output_lines.append(
                    f"• [<code>{e['time_str']}</code>] {icon} <b>[{e['currency']}] {e['event']}</b> "
                    f"(Est: <code>{e['estimate']}</code> | Prev: <code>{e['previous']}</code>)"
                )
            output_lines.append("")

        return "\n".join(output_lines).strip(), has_tier1

    except Exception as e:
        return f"Calendar processing issue: {str(e)}", False

# ==========================================
# 4. AI MACRO ENGINE & PROMPT SYNTHESIS
# ==========================================
def generate_report(mode: str, calendar_data: str, intermarket_data: str, ohlc_prices: str, regime: str, regime_desc: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are the Chief Market Analyst at Apex Macro Desk. Create an easy-to-read, beginner-friendly **{mode.upper()} SESSION MARKET REPORT**.

WRITING & FORMATTING RULES:
1. **NO SHORTCODES:** Do NOT use abbreviations like "Comp 1", "Comp 8", or "Comp 1 & 3". Always write out the FULL, complete component names in plain, professional English.
2. **SIMPLE LANGUAGE:** Explain concepts in clear everyday English. Avoid dense Wall Street jargon.
3. **VISUAL SPACING:** Leave DOUBLE LINE BREAKS (`\n\n`) between sections, paragraphs, and individual asset breakdowns so the text is spacious and readable on phone screens.
4. **HTML STYLING:** Use <b>BOLD</b> for titles and tickers, <i>italics</i> for explanations, and <code>code tags</code> for prices/times. Do NOT use markdown asterisks (`*`) or unescaped `<` or `>` symbols.

SYSTEM INPUT DATA:
- **ACTIVE REGIME:** {regime} ({regime_desc})
- **LIVE PRICES (14 TARGET ASSETS):**
{ohlc_prices}
- **INTERMARKET INDICATORS:**
{intermarket_data}
- **TIMED SESSION NEWS DATA:**
{calendar_data}

REQUIRED TELEGRAM OUTPUT STRUCTURE:

🏛️ <b>APEX MACRO DESK — {mode.upper()} SESSION INTELLIGENCE</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>1. 🌐 DECISION ENGINE & MACRO FRAMEWORK OVERVIEW</b>

• <b>Active Market Regime:</b> <i>{regime} — {regime_desc}</i>

• <b>Interest Rate & Bond Yield Differentials:</b> <i>2 plain English sentences explaining how bond yield movements and real interest rate gaps are driving capital flows.</i>

• <b>Central Bank Policy Stance:</b> <i>2 plain English sentences explaining expectations for the Fed, ECB, and BOE.</i>

• <b>Growth & Macro Divergence:</b> <i>2 plain English sentences comparing economic performance across key regions.</i>

• <b>Market Sentiment & Volatility Dynamics:</b> <i>2 plain English sentences assessing fear levels, risk-on/risk-off sentiment, and VIX status.</i>

• <b>Institutional Order Flow & Positioning:</b> <i>2 plain English sentences on big money movements and scheduled news event risks for today.</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>2. 📅 SCHEDULED SESSION NEWS & VOLATILITY MAP</b>
{calendar_data}

━━━━━━━━━━━━━━━━━━━━━━

<b>3. 🥇 MAJOR FOREX PAIRS & DXY</b>
(Evaluate ONLY these 7 assets: DXY, EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF)

Format every single asset with double line spacing like this:

<b>EURUSD</b> 🟢 BULLISH | Price: <code>1.0850</code>
↳ <b>Macro Drivers:</b> <i>Write 1-2 plain English sentences explaining drivers based on interest rates and central bank stance.</i>

<b>GBPUSD</b> 🔴 BEARISH | Price: <code>1.2620</code>
↳ <b>Macro Drivers:</b> <i>Write 1-2 plain English sentences explaining drivers.</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>4. 📈 INDICES & COMMODITIES</b>
(Evaluate ONLY these 5 assets: GOLD, SILVER, SP500, NAS100, GER40)

Format each asset with space:

<b>GOLD</b> 🟢 BULLISH | Price: <code>2400.50</code>
↳ <b>Macro Drivers:</b> <i>Write 1-2 plain English sentences explaining market drivers.</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>5. 🪙 CRYPTO ASSETS</b>
(Evaluate ONLY these 2 assets: BTCUSDT, ETHUSDT)

Format each asset with space:

<b>BTCUSDT</b> ⚪ NEUTRAL | Price: <code>64200.00</code>
↳ <b>Macro Drivers:</b> <i>Write 1-2 plain English sentences explaining market drivers.</i>

━━━━━━━━━━━━━━━━━━━━━━

<b>6. 🎯 HIGH CONVICTION SETUPS</b>
(Pick the 2-3 best trade setups where interest rates, risk sentiment, and institutional flows all point in the same direction)

• <b>[ASSET TICKER] — [BUY / SELL]</b>
  - <b>Trigger Zone:</b> <code>[Price]</code>
  - <b>Target Price:</b> <code>[Price]</code> | <b>Stop Loss:</b> <code>[Price]</code>
  - <b>Trade Game Plan:</b> <i>1 plain English sentence explaining why interest rates, risk sentiment, and market flows align for this setup.</i>
"""

    model_candidates = [
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
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
# 5. BANNER IMAGE GENERATOR
# ==========================================
def generate_banner(mode: str, regime: str) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    ax.axis('off')

    title = "APEX MACRO DESK"
    subtitle = f"{mode.upper()} SESSION INTELLIGENCE — LIVE REPORT"
    regime_badge = f"ACTIVE REGIME: {regime}"
    date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d | %H:%M UTC")

    ax.text(0.5, 0.70, title, color='#58A6FF', fontsize=22, fontweight='bold', ha='center', va='center')
    ax.text(0.5, 0.48, subtitle, color='#FFFFFF', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(0.5, 0.28, regime_badge, color='#3FB950', fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(0.5, 0.12, date_str, color='#8B949E', fontsize=8, ha='center', va='center')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf

# ==========================================
# 6. TELEGRAM DISPATCH WITH CHUNKING
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
# 7. CLI EXECUTION ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Apex Macro Desk Decision System")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "london", "ny", "eod", "update"], help="Session mode")
    args = parser.parse_args()

    gemini_key = os.getenv("GEMINI_API_KEY")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    fmp_key = os.getenv("FMP_API_KEY", "")

    if not gemini_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    if not bot_token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables must be set.")
        sys.exit(1)

    modes_to_run = ["london", "ny", "eod", "update"] if args.mode == "all" else [args.mode]

    print("🧠 Running Regime Classifier...")
    regime, regime_desc, vix_val, yield_val = classify_market_regime()
    print(f"   ↳ Active Regime: {regime} | VIX: {vix_val:.2f} | US10Y: {yield_val:.2f}%")

    print("📊 Ingesting Market Prices (14 Target Assets)...")
    ohlc_prices = fetch_market_data()

    print("🌐 Ingesting Intermarket Data & Yields...")
    intermarket_data = fetch_intermarket_data()

    for current_mode in modes_to_run:
        print(f"\n🚀 Processing [{current_mode.upper()} SESSION]...")
        print("📅 Fetching & Sorting Session News Events by Volatility...")
        calendar_data, has_tier1 = fetch_economic_calendar(fmp_key, current_mode)

        print("🤖 Synthesizing Institutional Macro Report with Gemini...")
        report_text = generate_report(current_mode, calendar_data, intermarket_data, ohlc_prices, regime, regime_desc, gemini_key)

        print("🖼️ Generating Session Banner Graphic...")
        banner_buf = generate_banner(current_mode, regime)

        print("📡 Broadcasting Session Intelligence directly to Telegram...")
        success = send_telegram(bot_token, chat_id, report_text, banner_buf)

        if success:
            print(f"✅ [{current_mode.upper()}] Report broadcasted successfully to Telegram!")
        else:
            print(f"❌ [{current_mode.upper()}] Broadcast failed.")
            sys.exit(1)

if __name__ == "__main__":
    main()
