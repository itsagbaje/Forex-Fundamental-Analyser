import os
import re
import sys
import logging
import requests
from enum import Enum
from google import genai
from google.genai import types

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Initialize Gemini Client (requires GEMINI_API_KEY environment variable)
client = genai.Client()
MODEL_NAME = "gemini-3.5-flash"

WATCHLIST = [
    "EURUSD", "GBPUSD", "NZDUSD", "AUDUSD", "USDCHF", "USDCAD",
    "GOLD", "SILVER", "SP500", "NAS100", "GER40", "BTCUSDT", "ETHUSDT"
]

class ReportMode(Enum):
    LONDON_OUTLOOK = "LONDON_OUTLOOK"
    NEW_YORK_OUTLOOK = "NEW_YORK_OUTLOOK"
    POST_NEWS_UPDATE = "POST_NEWS_UPDATE"
    EOD_RECAP = "EOD_RECAP"


# System Prompts customized per trading phase
PROMPT_TEMPLATES = {
    ReportMode.LONDON_OUTLOOK: f"""
You are a Senior Forex Strategist. Generate today's LONDON SESSION OUTLOOK formatted for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED SECTIONS:
1. ASIAN SESSION RECAP & LONDON CALENDAR:
   - Asian session price range/liquidity sweeps and key London macroeconomic catalysts.
   - Global Risk Sentiment [RISK-ON / RISK-OFF / MIXED].
2. IN-DEPTH DXY MACRO ANALYSIS (ANCHOR):
   - Fundamental drivers, Treasury yields, technical structure, and explicit DXY Bias [BULLISH / BEARISH / NEUTRAL].
3. COMPLETE WATCHLIST DIRECTIONAL BIAS:
   - Directional bias for all 13 watchlist assets.
   - Append "[DECOUPLING NOTE: reason]" if an asset deviates from standard DXY inverse correlation.
4. TOP HIGH-CONVICTION SETUPS (MINIMUM 5):
   - AT LEAST 5 setups (Pair, Long/Short, Macro Driver, Entry Zone, Invalidation/Stop Loss, Target/Take Profit).
5. RISK DISCLAIMER:
   - Bold educational disclaimer requiring personal risk management.
""",

    ReportMode.NEW_YORK_OUTLOOK: f"""
You are a Senior Forex Strategist. Generate today's NEW YORK SESSION OUTLOOK formatted for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED SECTIONS:
1. LONDON SESSION MID-DAY RECAP & US CALENDAR:
   - Key moves during London open, upcoming US high-impact economic releases (CPI, NFP, FOMC, etc.).
   - Global Risk Sentiment shift heading into NY Overlap.
2. DXY MACRO ANCHOR UPDATE:
   - Mid-day DXY bias evaluation [BULLISH / BEARISH / NEUTRAL] reflecting US session yield activity.
3. COMPLETE WATCHLIST DIRECTIONAL BIAS:
   - Directional bias for all 13 watchlist assets.
   - Append "[DECOUPLING NOTE: reason]" if an asset deviates from standard DXY inverse correlation.
4. TOP HIGH-CONVICTION NY SETUPS (MINIMUM 5):
   - AT LEAST 5 setups tailored for the NY session / London Overlap volatility.
5. RISK DISCLAIMER:
   - Educational notice regarding high volume during NY overlap.
""",

    ReportMode.POST_NEWS_UPDATE: f"""
You are a Senior Forex Strategist. Generate a POST-NEWS ANALYSIS & REACTION REPORT formatted for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED SECTIONS:
1. EVENT BREAKDOWN & DATA OUTCOME:
   - Release details (e.g., CPI, NFP, Rate Decision) showing Actual vs. Forecast / Consensus.
2. IMMEDIATE DXY & YIELD REACTION:
   - Volatility spike, market structure shift, and updated DXY bias.
3. WATCHLIST BIAS UPDATES & DECOUPLING ADJUSTMENTS:
   - Detail which assets held structure vs. which invalidated levels.
   - Append "[DECOUPLING NOTE: reason]" for unexpected divergence.
4. TOP POST-NEWS HIGH-CONVICTION SETUPS (MINIMUM 5):
   - AT LEAST 5 setups targeting post-spike retests, fair value gaps, or continuation entries.
5. RISK WARNING:
   - Warning regarding post-release spread expansion and slippage.
""",

    ReportMode.EOD_RECAP: f"""
You are a Senior Forex Strategist. Generate an END-OF-DAY (EOD) SESSION RECAP formatted for Telegram.
Watchlist: {', '.join(WATCHLIST)}

REQUIRED SECTIONS:
1. DAILY SESSION SUMMARY:
   - Main market themes, DXY daily close structure, and overarching macro drivers.
2. HIGH-CONVICTION SETUPS PERFORMANCE REVIEW:
   - Transparent review of today's setups (Targets hit, stops hit, or trades still active).
3. KEY LEVELS TESTED & HELD:
   - Liquidity sweeps, daily highs/lows, and notable S/R holds across the watchlist.
4. OVERNIGHT & ASIAN SESSION PREVIEW:
   - Key catalysts for the upcoming Asian/London handoff.
5. DISCLAIMER: Standard risk management disclaimer.
"""
}


def generate_report(news_feed: str, mode: ReportMode) -> str:
    """Queries Gemini API using the prompt template matching the current session mode."""
    system_instruction = PROMPT_TEMPLATES[mode]
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(f"Session Market Feeds & Technical Data:\n{news_feed}")]
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        )
    )
    return response.text


def process_report_content(raw_report: str) -> tuple[str, bool, int]:
    """
    Parses output to:
    1. Reformat [DECOUPLING NOTE: ...] tags into highlighted Telegram text.
    2. Count high-conviction setups to ensure minimum threshold of 5 is met.
    """
    formatted_report = re.sub(
        r"\[DECOUPLING NOTE:\s*(.*?)\]",
        r"\n  ⚡ *[DECOUPLING NOTE]:* _\1_",
        raw_report,
        flags=re.IGNORECASE
    )

    setup_matches = re.findall(r"(?:^|\n)(?:\d+[\️⃣\.]|\d+\)\s*)", formatted_report)
    setup_count = len(setup_matches)
    is_valid = setup_count >= 5

    return formatted_report, is_valid, setup_count


def generate_report_with_retry(news_feed: str, mode: ReportMode, max_retries: int = 3) -> tuple[str, bool, int]:
    """Retries generation automatically if fewer than 5 setups are produced."""
    current_feed = news_feed

    for attempt in range(1, max_retries + 1):
        logging.info(f"[{mode.value}] Attempt {attempt}/{max_retries}: Requesting generation from Gemini...")
        
        raw_output = generate_report(current_feed, mode)
        formatted_text, meets_min_setups, total_setups = process_report_content(raw_output)

        # EOD Recap evaluates setup outcomes so setup threshold validation is skipped
        if meets_min_setups or mode == ReportMode.EOD_RECAP:
            logging.info(f"[{mode.value}] Success! Generated {total_setups} setups/entries.")
            return formatted_text, True, total_setups
        
        logging.warning(f"[{mode.value}] Attempt {attempt} generated only {total_setups}/5 setups.")

        if attempt < max_retries:
            current_feed += (
                f"\n\n[SYSTEM DIRECTIVE - RETRY {attempt + 1}]: "
                f"Your previous attempt produced only {total_setups} setups. "
                f"You MUST generate AT LEAST 5 high-conviction setups."
            )

    logging.error(f"[{mode.value}] Exceeded max retries. Using latest output ({total_setups} setups).")
    return formatted_text, False, total_setups


def send_telegram_chunked(bot_token: str, chat_id: str, text: str):
    """Safely splits long messages along line breaks to respect Telegram's 4,096 char limit."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    MAX_CHUNK_SIZE = 4000

    if len(text) <= MAX_CHUNK_SIZE:
        chunks = [text]
    else:
        chunks = []
        current_chunk = ""
        for line in text.splitlines(keepends=True):
            if len(current_chunk) + len(line) > MAX_CHUNK_SIZE:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += line
        if current_chunk:
            chunks.append(current_chunk)

    for idx, chunk in enumerate(chunks, 1):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        res = requests.post(url, json=payload)
        res.raise_for_status()
        logging.info(f"Delivered chunk {idx}/{len(chunks)} to Telegram chat {chat_id}")


def send_admin_alert(bot_token: str, admin_chat_id: str, message: str):
    """Sends failure diagnostics to the admin chat."""
    if not bot_token or not admin_chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": admin_chat_id,
        "text": f"🚨 *PIPELINE ALERT* 🚨\n\n{message}",
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logging.error(f"Failed to send admin alert: {e}")


def main():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHANNEL_ID")
    admin_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID")

    if not bot_token or not chat_id:
        logging.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHANNEL_ID environment variables.")
        sys.exit(1)

    # SELECT SESSION MODE:
    # Option 1: ReportMode.LONDON_OUTLOOK
    # Option 2: ReportMode.NEW_YORK_OUTLOOK
    # Option 3: ReportMode.POST_NEWS_UPDATE
    # Option 4: ReportMode.EOD_RECAP
    
    # Example reading mode from command-line arg (e.g. `python main.py LONDON_OUTLOOK`) or fallback
    mode_str = sys.argv[1] if len(sys.argv) > 1 else "LONDON_OUTLOOK"
    try:
        active_mode = ReportMode[mode_str]
    except KeyError:
        logging.warning(f"Invalid mode '{mode_str}'. Defaulting to LONDON_OUTLOOK.")
        active_mode = ReportMode.LONDON_OUTLOOK

    # Sample input feed (Replace with your dynamic scraper feed)
    sample_feed = """
    SESSION DATA INPUT:
    - DXY holding 103.80 after mild Asian consolidation.
    - UK Manufacturing PMI beat expectations at 51.2 vs 49.8 forecast.
    - Crude Oil bouncing off $72.50 support.
    - US 10Y Yields steady at 4.14%.
    - Gold retesting $2,030 demand zone following Asian liquidity sweep.
    """

    logging.info(f"Running pipeline for mode: {active_mode.value}")

    report_text, validation_passed, setup_count = generate_report_with_retry(
        news_feed=sample_feed,
        mode=active_mode,
        max_retries=3
    )

    if not validation_passed and active_mode != ReportMode.EOD_RECAP:
        send_admin_alert(
            bot_token, 
            admin_chat_id, 
            f"Mode {active_mode.value} produced only {setup_count}/5 setups after 3 retries."
        )

    try:
        send_telegram_chunked(bot_token, chat_id, report_text)
        logging.info(f"Successfully published {active_mode.value} report to Telegram.")
    except Exception as e:
        logging.error(f"Telegram delivery failed: {e}")
        send_admin_alert(bot_token, admin_chat_id, f"Execution error in {active_mode.value}: {e}")


if __name__ == "__main__":
    main()
