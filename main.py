import os
import sys
import argparse
import io
import requests
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from google import genai

# ==========================================
# 1. ASSET WATCHLIST & CATEGORIZATION
# ==========================================
WATCHLIST_TICKERS = {
    # 🥇 Majors & Gold
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "GOLD": "GC=F",
    # 🪙 Crypto Assets
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    # 📈 Crosses & Indices
    "GBPJPY": "GBPJPY=X",
    "EURJPY": "EURJPY=X",
    "EURCAD": "EURCAD=X",
    "GBPCAD": "GBPCAD=X",
    "SILVER": "SI=F",
    "GER40": "^GDAXI",
    "NAS100": "NQ=F",
    "SP500": "ES=F",
    "JP225": "^N225"
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
            desc = f"High Volatility Environment (VIX: {cur_vix:.2f}, 5d Slope: {vix_5d_pct:+.1f}%). Safe-haven flows and sentiment override fundamentals."
        elif us10y_daily_move > 0.05:
            regime = "RATE_DRIVEN"
            desc = f"Active Yield Movements (US10Y at {cur_10y:.2f}%). Real interest rate differentials drive capital flows."
        else:
            regime = "GROWTH_DIVERGENCE"
            desc = f"Calm Volatility Environment (VIX: {cur_vix:.2f}). Relative economic performance and PMI expectations drive asset pairs."

        return regime, desc, cur_vix, cur_10y
    except Exception as e:
        return "RATE_DRIVEN", f"Default fallback regime (Data error: {str(e)})", 18.0, 4.25

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
                price_lines.append(f"• {name}: Price = {latest_price:.4f} (24h: {change_pct:+.2f}%)")
            else:
                price_lines.append(f"• {name}: Live Price unavailable")
    except Exception as e:
        price_lines.append(f"Error fetching prices: {str(e)}")
    return "\n".join(price_lines)

def fetch_intermarket_data():
    lines = []
    try:
        data = yf.download(tickers=list(INTERMARKET_TICKERS.values()), period="5d", interval="1d", progress=False)['Close']
        for name, ticker in INTERMARKET_TICKERS.items():
            if ticker in data and not data[ticker].dropna().empty:
                val = data[ticker].dropna().iloc[-1]
                lines.append(f"{name}: {val:.3f}")
    except Exception as e:
        lines.append(f"Intermarket data fetch error: {str(e)}")
    return " | ".join(lines) if lines else "DXY & Yields temporarily unavailable"

def fetch_economic_calendar(api_key):
    if not api_key:
        return "Economic calendar API key missing. Operating on intermarket yields.", False
    
    today = datetime.date.today().strftime("%Y-%m-%d")
    url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today}&to={today}&apikey={api_key}"
    tier1_keywords = ["Interest Rate", "Fed", "ECB", "BOE", "BOJ", "CPI", "Inflation", "Non Farm", "NFP", "GDP"]
    has_tier1_today = False
    
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            events = res.json()
            high_impact = [e for e in events if e.get("impact") in ["High", "Medium"] and e.get("currency") in ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD"]]
            if not high_impact:
                return "No Tier 1 or Tier 2 economic events scheduled today.", False
            
            tier1_lines, tier2_lines = [], []
            for ev in high_impact[:8]:
                ev_name = ev.get('event', '')
                is_t1 = any(kw.lower() in ev_name.lower() for kw in tier1_keywords) or ev.get('impact') == 'High'
                line = f"• [{ev.get('currency')}] {ev_name} (Est: {ev.get('estimate', 'N/A')}, Prev: {ev.get('previous', 'N/A')})"
                if is_t1:
                    tier1_lines.append(line)
                    has_tier1_today = True
                else:
                    tier2_lines.append(line)
            
            output = "<b>TIER 1 EVENT RISKS TODAY:</b>\n" + ("\n".join(tier1_lines) if tier1_lines else "None")
            output += "\n\n<b>TIER 2 SECONDARY DATA:</b>\n" + ("\n".join(tier2_lines) if tier2_lines else "None")
            return output, has_tier1_today
    except Exception as e:
        return f"Calendar fetch failed: {str(e)}", False
    return "No major calendar events available.", False

# ==========================================
# 3. AI MACRO ENGINE & PROMPT SYNTHESIS
# ==========================================
def generate_report(mode: str, calendar_data: str, intermarket_data: str, ohlc_prices: str, regime: str, regime_desc: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
You are the Lead Macro Strategist at Apex Macro Desk. Generate an educational, institutional-grade **{mode.upper()} Session Intelligence Report**.

SYSTEM ENVIRONMENT & DECISION ENGINE PARAMETERS:
- **CURRENT MARKET REGIME (Component 8):** {regime}
- **REGIME CONTEXT:** {regime_desc}
- **LIVE MARKET PRICES (ALL 18 ASSETS):**
{ohlc_prices}
- **INTERMARKET YIELDS & VOLATILITY:**
{intermarket_data}
- **EVENT RISK HIERARCHY (Component 7):**
{calendar_data}

**10-COMPONENT MACRO FRAMEWORK EVALUATION INSTRUCTIONS:**
1. **RATE DIFFERENTIALS (Component 1):** Evaluate real rate gaps (nominal yield minus inflation) between currency pairs.
2. **GROWTH DIFFERENTIALS (Component 2):** Assess relative GDP, PMI, and economic surprise index expectations.
3. **CB REACTION FUNCTION (Component 3):** Gauge moves relative to what markets have priced in vs central bank stance.
4. **RISK REGIME (Component 4):** Weight analysis based on active regime ({regime}). If RISK_DRIVEN, prioritize safe-haven flows; if RATE_DRIVEN, prioritize yield differentials.
5. **POSITIONING & FLOWS (Component 5):** Factor institutional COT net bias where applicable.
6. **RELATIVE SCORING RULE (Component 9):** Score factors as Currency A vs Currency B (relative divergence), never as absolute numbers.
7. **CONVICTION RULE (Component 10):** Flag high conviction ONLY when Rates, Risk, and Flows agree. If factors conflict, assign ⚪ NEUTRAL / NO TRADE.

**TONE & READABILITY:**
- Explain the "WHY" behind every single bias in friendly, conversational, educational English (1-2 sentences per asset).
- Use ONLY Telegram HTML tags: <b>bold</b>, <i>italics</i>, and <code>code</code>. Do NOT use markdown asterisks (*) or raw '<' and '>' characters.

**REQUIRED OUTPUT STRUCTURE:**

🏛️ <b>APEX MACRO DESK — {mode.upper()} SESSION INTELLIGENCE</b>
━━━━━━━━━━━━━━━━━━━━━━

1. <b>🌐 DECISION ENGINE & MARKET REGIME</b>
• <b>Active Regime:</b> <i><b>{regime}</b> — {regime_desc}</i>
• <b>Dollar & Rates:</b> <i>2 friendly sentences explaining DXY and Treasury yields in plain English.</i>
• <b>Market Vibe:</b> <i>2 friendly sentences explaining overall risk sentiment.</i>

2. <b>🥇 MAJORS & GOLD (PRIMARY FOCUS)</b>
(Must evaluate ALL 7 assets in this exact order: EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, GOLD)
Format each asset as:
<b>[TICKER]</b> [🟢 BULLISH / 🔴 BEARISH / ⚪ NEUTRAL] | Level: <code>[Live Price]</code>
↳ <b>Why?</b> <i>1-2 friendly sentences explaining the exact macro reasoning (yield gaps, CB stance, risk flows) behind this bias.</i>

3. <b>🪙 CRYPTO ASSETS</b>
(Must evaluate ALL 2 assets in this exact order: BTCUSD, ETHUSD)
Format each same as above.

4. <b>📈 CROSSES & INDICES</b>
(Must evaluate ALL 9 assets in this exact order: GBPJPY, EURJPY, EURCAD, GBPCAD, SILVER, GER40, NAS100, SP500, JP225)
Format each same as above.

5. <b>🎯 APEX SESSION TRIGGER MATRIX</b>
• Identify the <b>3 Highest-Conviction Setups</b> where at least 3 independent framework factors align.
Format as:
  - <b>[ASSET]</b> [BUY/SELL]
    • <b>Trigger Level:</b> <code>[Price]</code>
    • <b>Target:</b> <code>[Price]</code> | <b>Invalidation:</b> <code>[Price]</code>
    • <b>Game Plan:</b> <i>1 sentence explaining why this trade aligns with current macro differentials.</i>
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ==========================================
# 4. BANNER IMAGE GENERATOR
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
# 5. DISPATCH TO TELEGRAM
# ==========================================
def send_telegram(bot_token: str, chat_id: str, message: str, image_buf: io.BytesIO):
    photo_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {'photo': ('banner.png', image_buf, 'image/png')}
    
    if len(message) > 1000:
        requests.post(photo_url, data={'chat_id': chat_id}, files=files, timeout=15)
        msg_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        res = requests.post(msg_url, json=payload, timeout=15)
    else:
        payload = {
            'chat_id': chat_id,
            'caption': message,
            'parse_mode': 'HTML'
        }
        res = requests.post(photo_url, data=payload, files=files, timeout=15)
        
    return res.status_code == 200

# ==========================================
# 6. CLI EXECUTION ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Apex Macro Desk 10-Component Decision System")
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

    print("🧠 Running Regime Classifier (Component 8)...")
    regime, regime_desc, vix_val, yield_val = classify_market_regime()
    print(f"   ↳ Active Regime: {regime} | VIX: {vix_val:.2f} | US10Y: {yield_val:.2f}%")
    
    print("📊 Ingesting Market Prices (18 Assets)...")
    ohlc_prices = fetch_market_data()
    
    print("🌐 Ingesting Intermarket Data & Yields...")
    intermarket_data = fetch_intermarket_data()
    
    print("📅 Fetching Economic Calendar & Evaluating Event Risk Hierarchy...")
    calendar_data, has_tier1 = fetch_economic_calendar(fmp_key)

    for current_mode in modes_to_run:
        print(f"\n🚀 Processing [{current_mode.upper()} SESSION]...")
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
