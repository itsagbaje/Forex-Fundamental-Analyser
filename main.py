import os
import re
import sys
import io
import time
import logging
import argparse
import requests

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
    "EURUSD", "GBPUSD", "NZDUSD", "AUDUSD", "USDCHF", "USDCAD",
    "GOLD", "SILVER", "SP500", "NAS100", "DXY", "BTCUSDT", "ETHUSDT", "GER40"
]

PREFERRED_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
TELEGRAM_ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MANDATORY_DISCLAIMER = """───────────────────────────
⚠️ *RISK DISCLAIMER & SUBSCRIBER NOTICE*
_The setups and directional biases provided above are analytical insights for educational purposes only. Do NOT blindly copy or execute these ideas. Trading requires independent trade confirmation, strict risk management, appropriate position sizing, and adherence to stated invalidation levels._"""

# ---------------------------------------------------------------------------
# System Instructions & Telegram Formatting Rules
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = f"""
You are a Lead Institutional Macro & Forex Fundamental Analyst.
You analyze global financial markets across this complete watchlist:
{', '.join(WATCHLIST)}.

TELEGRAM VISUAL FORMATTING RULES (CRITICAL):
1. DO NOT use `#` or `##` Markdown headers (Telegram displays them as raw text).
2. Use Emoji + Bold ALL-CAPS for section headers (e.g., `📌 *DEDICATED DXY MACRO ANALYSIS*`).
3. Use `───────────────────────────` as a divider line between major sections.
4. Use `•` bullet points with bold field names for structured data (e.g., `• *Macro Catalyst:* CPI beats consensus`).
5. Keep paragraphs short (2-3 sentences max) with clean line breaks for effortless mobile reading.

CORE ANALYTICAL PRINCIPLES:
1. DEDICATED DXY ANCHOR: DXY must have its own dedicated section as the primary reference for USD pairs. However, NEVER treat it as a mechanical inverse indicator. Account for regional currency relative strength/weakness.
2. INTRADAY MATERIAL CHANGE GATEKEEPER: When evaluating intraday market updates or news events, FIRST assess if there is a genuine material shift in DXY bias, asset biases, or setups.
   - IF NO MATERIAL CHANGE HAS OCCURRED: Start your entire response with `[NO_MATERIAL_CHANGE]` followed by a brief 1-2 sentence explanation of why the current thesis holds. Do NOT generate a full report.
   - IF A MATERIAL CHANGE HAS OCCURRED: Start your response with `[MATERIAL_CHANGE]` and proceed with the full Intraday Update Report format.
3. WATCHLIST DIRECTIONAL BIAS: Provide an explicit directional bias (🟢 Bullish / 🔴 Bearish / 🟡 Neutral) for EVERY asset on the watchlist.
4. RANKED HIGH-CONVICTION SETUPS (MINIMUM 5): Generate at least 5 ranked setups. You MUST use the exact live market prices provided in the prompt context. Each setup must follow this exact format:

   *Setup #X: [ASSET] — [BUY / SELL]*
   • *Macro Catalyst:* <Context>
   • *DXY / Correlation Stance:* <Context>
   • *COT & Retail Sentiment:* <Context>
   • *Entry / Retest Zone:* <Price level based on live feed>
   • *Target Zone:* <Price level>
   • *Invalidation Level:* <Price level>
   • *Timing Window:* <Session timing>
"""

PROMPT_TEMPLATES = {
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
Explicit Directional Bias (🟢 Bullish / 🔴 Bearish / 🟡 Neutral) and 1-sentence macro driver for ALL 14 assets:
{', '.join(WATCHLIST)}.

───────────────────────────
🔄 *INTRADAY CONTINUITY & THESIS CHECK*
Prevailing macro narrative and adjustments from overnight Asian flows.

───────────────────────────
🎯 *TOP HIGH-CONVICTION SETUPS (RANKED #1 TO #5 MINIMUM)*
Provide AT LEAST 5 ranked setups matching the live price data feed.

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
US Dollar Index positioning heading into Wall Street open, bond yield momentum, and Fed pricing.

───────────────────────────
📊 *COMPLETE WATCHLIST DIRECTIONAL BIAS MATRIX*
Explicit Directional Bias (🟢 Bullish / 🔴 Bearish / 🟡 Neutral) for ALL 14 assets:
{', '.join(WATCHLIST)}.

───────────────────────────
🔄 *INTRADAY CONTINUITY & THESIS TRANSITION*
Compare London price action against the morning thesis. Explicitly state: What changed during London, why it changed, and how it impacts New York flows.

───────────────────────────
🎯 *TOP HIGH-CONVICTION SETUPS (RANKED #1 TO #5 MINIMUM)*
AT LEAST 5 ranked setups tailored for NY session volatility using live asset pricing.

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
Shift in DXY structure, bond yields, and interest rate expectations.

───────────────────────────
📊 *UPDATED DIRECTIONAL BIAS MATRIX*
Post-event bias for ALL 14 assets: {', '.join(WATCHLIST)}.

───────────────────────────
🎯 *UPDATED HIGH-CONVICTION SETUPS (RANKED #1 TO #5 MINIMUM)*
AT LEAST 5 updated setups reflecting the new thesis using live market prices.

───────────────────────────
⚠️ *RISK WARNING & LIQUIDITY NOTES*
Slippage, spread expansion, and session liquidity advisories.
""",

    "EOD_RECAP": f"""
Generate the **END OF DAY MACRO & FUNDAMENTAL RECAP** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

🌆 *END OF DAY MACRO & FUNDAMENTAL RECAP*
───────────────────────────

💵 *DEDICATED DXY & CROSS-ASSET RECAP*
In-depth review of DXY performance across Asian, London, and NY sessions alongside Treasury yield behavior.

───────────────────────────
📊 *DAILY CLOSING BIAS MATRIX*
Closing price action and directional bias for ALL 14 assets: {', '.join(WATCHLIST)}.

───────────────────────────
🔄 *DAILY THESIS REVIEW & DECOUPLING SUMMARY*
Summary of key market drivers, unexpected decoupling events, and structural trend shifts observed today.

───────────────────────────
📅 *TOMORROW'S CALENDAR & PRELIMINARY OUTLOOK*
Key high-impact events for upcoming sessions and early directional context.
"""
}

# ---------------------------------------------------------------------------
# Live Data Fetcher Helper
# ---------------------------------------------------------------------------
def fetch_live_market_data() -> str:
    ticker_mapping = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "NZDUSD": "NZDUSD=X",
        "AUDUSD": "AUDUSD=X", "USDCHF": "USDCHF=X", "USDCAD": "USDCAD=X",
        "GOLD": "GC=F", "SILVER": "SI=F", "SP500": "^GSPC", "NAS100": "^NDX",
        "DXY": "DX-Y.NYB", "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "GER40": "^GDAXI"
    }

    price_lines = []
    logging.info("Fetching real-time market prices from Yahoo Finance...")

    for asset, symbol in ticker_mapping.items():
        try:
            t = yf.Ticker(symbol)
            todays_data = t.history(period="1d")
            if not todays_data.empty:
                current_price = todays_data['Close'].iloc[-1]
                price_lines.append(f"- {asset}: {current_price:.4f}" if current_price < 10 else f"- {asset}: {current_price:.2f}")
            else:
                price_lines.append(f"- {asset}: Data unavailable")
        except Exception as e:
            logging.warning(f"Could not fetch live price for {asset}: {e}")
            price_lines.append(f"- {asset}: N/A")

    return "\n".join(price_lines)

# ---------------------------------------------------------------------------
# Visual Banner Generator
# ---------------------------------------------------------------------------
def generate_session_banner(mode: str) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=150)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    display_titles = {
        "LONDON_OUTLOOK": "LONDON SESSION OUTLOOK",
        "NEW_YORK_OUTLOOK": "NEW YORK SESSION OUTLOOK",
        "INTRADAY_UPDATE": "INTRADAY THESIS UPDATE",
        "EOD_RECAP": "END-OF-DAY MACRO RECAP"
    }
    subtitle = display_titles.get(mode, "MARKET INTELLIGENCE BRIEF")

    ax.text(0.05, 0.70, "APEX INSTITUTIONAL MACRO DESK", fontsize=11, color='#8b949e', fontweight='bold', fontfamily='monospace')
    ax.text(0.05, 0.40, subtitle, fontsize=22, color='#ffffff', fontweight='bold', fontfamily='sans-serif')
    
    timestamp_str = time.strftime("%A, %d %b %Y | %H:%M UTC", time.gmtime())
    ax.text(0.05, 0.15, f"GENERATED: {timestamp_str}", fontsize=9, color='#58a6ff', fontweight='bold', fontfamily='monospace')

    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

# ---------------------------------------------------------------------------
# Helpers & Validation
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

def send_telegram_media_and_message(text: str, banner_buf: io.BytesIO) -> bool:
    photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    msg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # 1. Send Visual Banner Image
    try:
        files = {'photo': ('banner.png', banner_buf.getvalue(), 'image/png')}
        photo_res = requests.post(photo_url, data={'chat_id': TELEGRAM_CHANNEL_ID, 'caption': "📊 *Apex Institutional Intelligence Briefing*"}, files=files, timeout=15)
        if photo_res.status_code != 200:
            logging.warning(f"Banner upload alert ({photo_res.status_code}): {photo_res.text}")
    except Exception as e:
        logging.warning(f"Banner photo dispatch exception: {e}")

    # 2. Send Chunked Text Report
    max_length = 4000
    chunks = []
    while len(text) > max_length:
        split_idx = text.rfind("\n", 0, max_length)
        if split_idx == -1: split_idx = max_length
        chunks.append(text[:split_idx])
        text = text[split_idx:].lstrip()
    chunks.append(text)

    success = True
    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHANNEL_ID,
            "text": chunk,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        res = requests.post(msg_url, json=payload)
        if res.status_code != 200:
            payload["parse_mode"] = ""
            res_fallback = requests.post(msg_url, json=payload)
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
    if matches: return len(set(matches))
    bullet_matches = re.findall(r"\n\s*(?:\d+\.|\*|\-)\s+\*Setup.*?(?:BUY|SELL|LONG|SHORT)", report_text, re.IGNORECASE)
    return len(bullet_matches)

def generate_report_with_retry(client: genai.Client, mode: str, custom_feed: str = None, max_retries: int = 5) -> str:
    prompt = PROMPT_TEMPLATES[mode]
    live_prices = fetch_live_market_data()
    prompt += f"\n\nLIVE MARKET PRICE FEED (MANDATORY REFERENCE FOR ALL PRICES):\n{live_prices}"

    if custom_feed:
        prompt += f"\n\nBREAKING EVENT FEED:\n{custom_feed}"

    report_text = ""
    setup_count = 0

    for attempt in range(1, max_retries + 1):
        logging.info(f"Calling Gemini ({PREFERRED_MODEL}) for mode '{mode}' (Attempt {attempt}/{max_retries})...")

        try:
            response = client.models.generate_content(
                model=PREFERRED_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS,
                    temperature=0.3,
                )
            )
            report_text = (response.text or "").strip()
        except Exception as e:
            logging.error(f"API Error during generate_content: {e}")
            if attempt < max_retries:
                sleep_duration = 5 * (2 ** (attempt - 1))
                logging.info(f"Temporary API failure. Retrying in {sleep_duration}s...")
                time.sleep(sleep_duration)
                continue
            else:
                raise e

        # Check for Intraday No Material Change flag
        if "[NO_MATERIAL_CHANGE]" in report_text:
            return "[NO_MATERIAL_CHANGE]"

        if not report_text.endswith(MANDATORY_DISCLAIMER.strip()):
            report_text += f"\n\n{MANDATORY_DISCLAIMER.strip()}"

        if mode == "EOD_RECAP":
            return report_text

        setup_count = count_setups(report_text)
        logging.info(f"Detected {setup_count} trade setup(s) in output.")

        if setup_count >= 5:
            return report_text

        logging.warning(f"Setup count validation failed ({setup_count}/5 required). Retrying...")
        prompt += "\n\nCRITICAL REQUIREMENT MISSING: You MUST generate AT LEAST 5 ranked, high-conviction trade setups clearly labeled (*Setup #1*, *Setup #2*, *Setup #3*, *Setup #4*, *Setup #5*) using the exact live prices provided."

    logging.warning("Max retries reached. Dispatching best output.")
    send_admin_alert(f"Mode `{mode}` generated only {setup_count}/5 setups after {max_retries} attempts.")
    return report_text

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    validate_environment()
    mode, custom_feed = parse_arguments()

    logging.info(f"Starting pipeline | Mode: {mode} | Model: {PREFERRED_MODEL}")

    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        report = generate_report_with_retry(client=client, mode=mode, custom_feed=custom_feed)

        # Gatekeeper check for suppressed intraday updates
        if report == "[NO_MATERIAL_CHANGE]":
            logging.info("Intraday reassessment completed: No material shift in macro thesis or setups. Telegram update suppressed.")
            sys.exit(0)

        # Clean tag if present
        report = report.replace("[MATERIAL_CHANGE]", "").strip()

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
