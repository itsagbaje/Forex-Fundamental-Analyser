import os
import re
import sys
import time
import logging
import argparse
import requests
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
# Telegram Visual Formatting & System Instructions
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = f"""
You are a Lead Institutional Macro & Forex Fundamental Analyst.
You analyze global financial markets across this complete watchlist:
{', '.join(WATCHLIST)}.

TELEGRAM VISUAL FORMATTING RULES (CRITICAL):
1. DO NOT use `#` or `##` Markdown headers (Telegram displays them as raw text).
2. Use Emoji + Bold ALL-CAPS for section headers (e.g., `📌 *DEDICATED DXY MACRO ANALYSIS*`).
3. Use `───────────────────────────` as a section divider line between major sections.
4. Use `•` bullet points with bold field names for structured data (e.g., `• *Macro Catalyst:* CPI beats consensus`).
5. Keep paragraphs short (2-3 sentences max) with clean line breaks for effortless mobile reading.

CORE ANALYTICAL PRINCIPLES:
1. DXY DEDICATED ANALYSIS: Treat DXY as the central macroeconomic anchor, but NEVER treat it as a mechanical inverse indicator. Recognize currency-specific relative strength/weakness (e.g., EURUSD can drop during DXY weakness if EUR weakness outpaces USD weakness). Explicitly explain these divergences.
2. CONTINUITY & TRANSITION: Maintain a coherent intraday thesis. When market conditions shift or prior assumptions break, explicitly explain WHAT changed, WHY it changed, which previous assumptions were invalidated, and the NEW bias.
3. WATCHLIST DIRECTIONAL BIAS: Every report must state an explicit directional bias (Bullish, Bearish, or Neutral/Ranging) for EVERY asset on the watchlist.
4. RANKED HIGH-CONVICTION SETUPS (MINIMUM 5): Generate at least 5 ranked setups for operational session outlooks and post-news reports. Each setup must follow this exact format:

   *Setup #X: [ASSET] — [BUY / SELL]*
   • *Macro Catalyst:* <Context>
   • *DXY / Correlation Stance:* <Context>
   • *COT & Retail Sentiment:* <Context>
   • *Technical Structure:* <MSS / CRT / FVG / OB details>
   • *Entry / Retest Zone:* <Price level>
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
Provide deep dive into DXY structure, Treasury yields, US interest rate expectations, and liquidity pools entering European hours. Explain any non-mechanical asset interactions.

───────────────────────────
📊 *DIRECTIONAL BIAS MATRIX*
List explicit Directional Bias (🟢 Bullish / 🔴 Bearish / 🟡 Neutral) and a 1-sentence macro driver for ALL 14 assets:
{', '.join(WATCHLIST)}.

───────────────────────────
🔄 *INTRADAY CONTINUITY & THESIS CHECK*
State the prevailing macro narrative. Note any overnight shifts or adjustments from prior sessions.

───────────────────────────
🎯 *TOP HIGH-CONVICTION SETUPS (RANKED #1 TO #5 MINIMUM)*
Provide AT LEAST 5 ranked setups using the exact setup layout defined in system instructions.

───────────────────────────
⚠️ *RISK WARNING & LIQUIDITY NOTES*
Key European data releases, central bank speak, and spread volatility warnings.
""",

    "NEW_YORK_OUTLOOK": f"""
Generate the **NEW YORK SESSION FUNDAMENTAL OUTLOOK** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

🏛 *NEW YORK SESSION FUNDAMENTAL OUTLOOK*
───────────────────────────

💵 *DEDICATED DXY MACRO ANALYSIS*
In-depth US Dollar Index stance heading into Wall Street open, bond yield momentum, and Fed expectations. Address relative strength dynamics vs majors, indices, and metals.

───────────────────────────
📊 *DIRECTIONAL BIAS MATRIX*
List explicit Directional Bias (🟢 Bullish / 🔴 Bearish / 🟡 Neutral) for ALL 14 assets:
{', '.join(WATCHLIST)}.

───────────────────────────
🔄 *INTRADAY CONTINUITY & THESIS TRANSITION*
Compare London price action against the original morning thesis. Explicitly state: What changed during London, why it changed, and how it impacts New York flows.

───────────────────────────
🎯 *TOP HIGH-CONVICTION SETUPS (RANKED #1 TO #5 MINIMUM)*
Provide AT LEAST 5 ranked setups tailored for NY session volatility using the exact setup layout defined in system instructions.

───────────────────────────
⚠️ *RISK WARNING & SPREAD ADVISORY*
Upcoming US economic releases and volatility alerts.
""",

    "POST_NEWS_UPDATE": f"""
Generate a **POST-NEWS ANALYSIS & REACTION REPORT** for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED VISUAL LAYOUT:

📰 *POST-NEWS FLASH ANALYSIS*
───────────────────────────

📊 *EVENT BREAKDOWN & DATA OUTCOME*
Release details (e.g., CPI, NFP, Rate Decisions). Compare actual vs consensus forecast.

───────────────────────────
💵 *DEDICATED DXY REACTION & YIELD IMPACT*
In-depth breakdown of immediate DXY structural shift, bond yield reaction, and repricing of central bank rate paths.

───────────────────────────
🔄 *BIAS REASSESSMENT & INVALIDATION LOG*
Explicitly answer:
• What changed in the macro picture?
• Which prior biases/assumptions are NO LONGER VALID?
• Which existing setups were invalidated by this event?
• What is the updated directional bias across the watchlist?

───────────────────────────
📊 *UPDATED DIRECTIONAL BIAS MATRIX*
Post-news bias for ALL 14 assets: {', '.join(WATCHLIST)}.

───────────────────────────
🎯 *TOP POST-NEWS HIGH-CONVICTION SETUPS (RANKED #1 TO #5 MINIMUM)*
AT LEAST 5 updated setups targeting post-news retests, FVG fills, or liquidity sweeps.

───────────────────────────
⚠️ *POST-RELEASE LIQUIDITY WARNING*
Slippage and secondary news wave advisories.
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
Review of performance and closing market structure for ALL 14 assets: {', '.join(WATCHLIST)}.

───────────────────────────
🔄 *DAILY THESIS REVIEW & DECOUPLING SUMMARY*
Summary of key market drivers, unexpected decoupling events, and structural trend shifts observed today.

───────────────────────────
📅 *TOMORROW'S CALENDAR & PRELIMINARY OUTLOOK*
Key high-impact events for upcoming sessions and early directional context.
"""
}

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
        "LONDON": "LONDON_OUTLOOK",
        "LONDON_OUTLOOK": "LONDON_OUTLOOK",
        "NEW_YORK": "NEW_YORK_OUTLOOK",
        "NEW_YORK_OUTLOOK": "NEW_YORK_OUTLOOK",
        "NY": "NEW_YORK_OUTLOOK",
        "POST_NEWS": "POST_NEWS_UPDATE",
        "POST_NEWS_UPDATE": "POST_NEWS_UPDATE",
        "EOD": "EOD_RECAP",
        "EOD_RECAP": "EOD_RECAP"
    }

    return mapping.get(cleaned, "LONDON_OUTLOOK"), args.feed

def validate_environment():
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHANNEL_ID:
        missing.append("TELEGRAM_CHANNEL_ID")

    if missing:
        logging.error(f"Missing required environment variable(s): {', '.join(missing)}")
        sys.exit(1)

def send_telegram_message(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_length = 4000

    chunks = []
    while len(text) > max_length:
        split_idx = text.rfind("\n", 0, max_length)
        if split_idx == -1:
            split_idx = max_length
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
        res = requests.post(url, json=payload)
        if res.status_code != 200:
            logging.warning(f"Markdown posting failed ({res.status_code}). Fallback to raw text...")
            payload["parse_mode"] = ""
            res_fallback = requests.post(url, json=payload)
            if res_fallback.status_code != 200:
                logging.error(f"Telegram API error: {res_fallback.text}")
                success = False
    return success

def send_admin_alert(message: str):
    if not TELEGRAM_ADMIN_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_ADMIN_CHAT_ID,
        "text": f"⚠️ *SYSTEM ALERT*\n{message}",
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logging.error(f"Admin alert dispatch failed: {e}")

def count_setups(report_text: str) -> int:
    matches = re.findall(r"(?i)(?:setup|rank|opportunity)\s*#?\d+", report_text)
    if matches:
        return len(set(matches))
    bullet_matches = re.findall(
        r"\n\s*(?:\d+\.|\*|\-)\s+\*Setup.*?(?:BUY|SELL|LONG|SHORT)", report_text, re.IGNORECASE
    )
    return len(bullet_matches)

def generate_report_with_retry(client: genai.Client, mode: str, custom_feed: str = None, max_retries: int = 5) -> str:
    prompt = PROMPT_TEMPLATES[mode]
    if custom_feed:
        prompt += f"\n\nLIVE MARKET DATA / BREAKING EVENT FEED:\n{custom_feed}"

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
                logging.info(f"Temporary API failure (503/429). Retrying in {sleep_duration} seconds...")
                time.sleep(sleep_duration)
                continue
            else:
                logging.error("Max API retries reached. Aborting pipeline.")
                raise e

        # Append mandatory disclaimer cleanly
        if not report_text.endswith(MANDATORY_DISCLAIMER.strip()):
            report_text += f"\n\n{MANDATORY_DISCLAIMER.strip()}"

        if mode == "EOD_RECAP":
            return report_text

        setup_count = count_setups(report_text)
        logging.info(f"Detected {setup_count} trade setup(s) in output.")

        if setup_count >= 5:
            return report_text

        logging.warning(f"Setup count validation failed ({setup_count}/5 required). Retrying with reinforced prompt...")
        prompt += "\n\nCRITICAL REQUIREMENT MISSING: You MUST generate AT LEAST 5 ranked, high-conviction trade setups clearly labeled (*Setup #1*, *Setup #2*, *Setup #3*, *Setup #4*, *Setup #5*)."

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
        logging.info("Report generation complete. Dispatching to Telegram...")

        if send_telegram_message(report):
            logging.info("Successfully published report to Telegram.")
        else:
            logging.error("Failed to post report to Telegram.")
            sys.exit(1)

    except Exception as e:
        error_msg = f"Pipeline execution failed for mode `{mode}`:\n`{str(e)}`"
        logging.error(error_msg, exc_info=True)
        send_admin_alert(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
    
