# tradie.v3.full.py
import os
import json
from datetime import datetime, timedelta
import numpy as np
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters,
    ContextTypes, ConversationHandler
)

# --- Gemini SDK ---
import google.generativeai as genai

# --- PostgreSQL ---
import psycopg2

# ========= CONFIG =========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_TOKEN")

EMBEDDING_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-2.5-flash"
SIM_THRESHOLD = 0.78
MAX_GROUPS_FOR_PROMPT = 20

# States for conversation
WAITING_FOR_MISTAKE = 1
WAITING_FOR_SEARCH = 2
WAITING_FOR_REMINDER = 3
WAITING_FOR_TREND = 4
# ==========================

# ---------- Gemini setup ----------
genai.configure(api_key=GEMINI_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)

# ---------- DB helpers (PostgreSQL version) ----------
def db_conn():
    return psycopg2.connect(os.environ["DATABASE_URL"], sslmode="require")

def init_db():
    with db_conn() as conn:
        c = conn.cursor()
        # existing mistakes table
        c.execute("""
            CREATE TABLE IF NOT EXISTS mistakes (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # new market_trends table
        c.execute("""
            CREATE TABLE IF NOT EXISTS market_trends (
                id SERIAL PRIMARY KEY,
                trend TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # new reminders table
        c.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

# Mistakes
def save_mistake(text: str, embedding: list[float]):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO mistakes (text, embedding, created_at) VALUES (%s, %s, %s)",
            (text, json.dumps(embedding), datetime.utcnow().isoformat())
        )
        conn.commit()

def load_mistakes():
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, text, embedding, created_at FROM mistakes ORDER BY id ASC")
        rows = c.fetchall()
    records = []
    for rid, text, emb_json, created_at in rows:
        try:
            vec = np.array(json.loads(emb_json), dtype=float)
            records.append({"id": rid, "text": text, "vec": vec, "created_at": created_at})
        except Exception:
            continue
    return records

# Reminders
def save_reminder(text: str):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO reminders (text, created_at) VALUES (%s, %s)",
            (text, datetime.utcnow().isoformat())
        )
        conn.commit()

def load_recent_reminders(days: int = 7):
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, text, created_at FROM reminders WHERE created_at >= %s ORDER BY id DESC", (cutoff,))
        rows = c.fetchall()
    return [{"id": r[0], "text": r[1], "created_at": r[2]} for r in rows]

def load_all_reminders():
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, text, created_at FROM reminders ORDER BY id DESC")
        rows = c.fetchall()
    return [{"id": r[0], "text": r[1], "created_at": r[2]} for r in rows]

# Market trends
def save_trend(trend_text: str):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO market_trends (trend, created_at) VALUES (%s, %s)",
            (trend_text.lower(), datetime.utcnow().isoformat())
        )
        conn.commit()

def get_latest_trend():
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT trend, created_at FROM market_trends ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
    if row:
        return {"trend": row[0], "created_at": row[1]}
    return None

# ---------- Embeddings & similarity ----------
def embed_text(text: str) -> list[float]:
    resp = genai.embed_content(model=EMBEDDING_MODEL, content=text)
    return resp["embedding"]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def group_by_similarity(records: list[dict], threshold: float = SIM_THRESHOLD):
    groups = []
    used = set()

    for i, r in enumerate(records):
        if r["id"] in used:
            continue
        seed = r
        group_members = [seed["text"]]
        group_ids = [seed["id"]]
        used.add(seed["id"])

        for j in range(i + 1, len(records)):
            s = records[j]
            if s["id"] in used:
                continue
            sim = cosine_sim(seed["vec"], s["vec"])
            if sim >= threshold:
                group_members.append(s["text"])
                group_ids.append(s["id"])
                used.add(s["id"])

        groups.append({"members": group_members, "ids": group_ids})

    for g in groups:
        g["label"] = min(g["members"], key=len)
        g["count"] = len(g["members"])

    groups.sort(key=lambda g: g["count"], reverse=True)
    return groups

# ---------- Gemini prompts (trend-aware) ----------
PROMPT_CHECKLIST = """You are a strict, concise trading mentor for a retail crypto trader.

Input:
Grouped mistakes (label -> count):
{group_lines}

Personal reminders (one per line):
{reminders}

Latest market trend: {latest_trend}
Today is {date}.

Tasks (in order):
1) List the repeating mistakes from most frequent to least, each on its own line with the count in parentheses.
2) Produce a PRE-TRADE CHECKLIST of 3–5 bullet points that directly prevent the top repeating mistakes and are adapted to the Latest market trend.

Trend guidance:
- If bullish → prefer trend-following, use momentum confirmation, wider stops on strong moves.
- If sideways → prefer range rules, tighter stops, avoid breakout chasing.
- If bearish → prioritize risk control, prefer short or avoid long bias, tighten position size.

Rules:
- Be ultra concise and practical (no lecturing).
- Checklist must be 3–5 bullets only, use "•" for bullets, one short sentence per bullet.
- Total output max ~120 words.
- If there are personal reminders, integrate any that directly reduce top mistakes.
- If there are no past mistakes, output a short default 3-bullet checklist.

Output format:
First the list of mistakes (one per line), then a blank line, then "PRE-TRADE CHECKLIST" and the bullets.
"""

PROMPT_WEEKLY = """You are a concise trading coach. Create a weekly summary for the date range {start_date} to {end_date}.

Inputs:
Top grouped mistakes (label -> count):
{group_lines}

Latest market trend: {latest_trend}
Personal reminders (one per line):
{reminders}

Task:
Write a 100–200 word weekly summary that includes:
- A one-line headline summarizing overall performance or theme.
- A short list of the top 3 repeating mistakes (label with counts).
- Two to three actionable recommendations tailored to the latest market trend and the user's reminders.
- A one-line closing action (what the user should focus on next week).

Tone: factual, non-judgmental, practical.
Length: 100–200 words. Use short sentences and clear action verbs.
"""

PROMPT_SEARCH = """You are a crypto/finance research assistant. The user asked: "{query}"

Task:
Produce a concise 100–200 word briefing focused on practical trading/market implications.
Structure:
- 1 sentence direct answer/headline.
- 2–3 short bullets highlighting the most relevant facts or drivers.
- 1 final short action item for traders (what to do right now).

Rules:
- Keep it factual and avoid speculation.
- If you cannot access live web information, begin with: "I cannot access live web sources; based on knowledge I have, ..." and then provide the best answer.
- Do not exceed 200 words.
"""

PROMPT_NEWSES = """You are a crypto news assistant. Today is {date}.

Task:
1. Consider top crypto/finance news sources.
2. Pick the 5 most important crypto news items published on {date}.
3. For each item, output:
   - Date (use {date})
   - Headline
   - ~100-word summary focused on trading/market implications.

Rules:
- Exactly 5 items only.
- Each summary ~100 words (80–120 words).
- Order from most important → least.
- Keep language factual and trader-oriented.
- If you cannot access live web information, be explicit: start the output with "I cannot access live web sources; based on knowledge I have, ..." and then provide best-effort summaries.
"""

# ---------- Gemini call wrapper with safe fallback ----------
def call_gemini(prompt: str, max_output_tokens: int = 400) -> str:
    try:
        resp = gen_model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            raise ValueError("Empty response")
        return text
    except Exception:
        # fallback short message
        return None

def gemini_or_fallback(prompt: str, fallback_text: str) -> str:
    out = call_gemini(prompt)
    if out:
        return out
    return fallback_text

# ---------- Helpers to build prompt inputs ----------
def top_group_lines_for_prompt(groups: list[dict], limit: int = 20):
    trimmed = groups[:limit]
    lines = [f"- {g['label']} -> {g['count']}" for g in trimmed]
    return "\n".join(lines)

def reminders_lines_for_prompt(reminders_list):
    if not reminders_list:
        return ""
    return "\n".join([f"• {r['text']}" for r in reminders_list])

# ---------- Handlers: new commands ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    init_db()
    # News
    news = fetch_news_brief()
    await update.message.reply_text("📢 NEWS FLASH + FUNNY NOTE\n" + "="*40 + f"\n{news}")

    # Checklist (trend-aware)
    records = load_mistakes()
    groups = group_by_similarity(records, SIM_THRESHOLD)
    # get recent reminders and latest trend
    latest_trend = get_latest_trend()
    latest_trend_text = latest_trend["trend"] if latest_trend else "unknown"
    reminders = load_recent_reminders(days=30)  # include recent reminders
    group_lines = top_group_lines_for_prompt(groups)
    rem_lines = reminders_lines_for_prompt(reminders)

    prompt = PROMPT_CHECKLIST.format(
        group_lines=group_lines,
        reminders=rem_lines,
        latest_trend=latest_trend_text,
        date=datetime.utcnow().strftime("%Y-%m-%d")
    )
    suggestions = gemini_or_fallback(prompt, generate_suggestions(groups))
    await update.message.reply_text("📋 PRE-TRADE CHECKLIST\n" + "="*40 + f"\n{suggestions}")

    # Ask for mistake
    await update.message.reply_text("\nHave you made any new mistakes? Reply here (or type 'skip').")
    return WAITING_FOR_MISTAKE

# ---------- Mistake handler ----------
async def handle_mistake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.lower() == "skip":
        await update.message.reply_text("👍 Skipped. Type /start anytime to run again.")
        return ConversationHandler.END
    try:
        emb = embed_text(text)
        save_mistake(text, emb)
        await update.message.reply_text("✅ Saved! Type /start anytime to run again.")
    except Exception as e:
        await update.message.reply_text(f"❌ Could not save mistake: {e}")
    return ConversationHandler.END

# ---------- /search flow ----------
async def search_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 What do you want to search for? (crypto/finance topic)")
    return WAITING_FOR_SEARCH

async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    prompt = PROMPT_SEARCH.format(query=query)
    resp = call_gemini(prompt)
    if not resp:
        resp = "❌ Gemini unavailable. Try again later."
    await update.message.reply_text(f"📢 Gemini Search Result:\n\n{resp}")
    return ConversationHandler.END

# ---------- /reminder flow ----------
async def reminder_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # if user provided argument with command, save immediately
    args = context.args
    if args:
        text = " ".join(args).strip()
        save_reminder(text)
        await update.message.reply_text("✅ Reminder saved.")
        return ConversationHandler.END
    await update.message.reply_text("✍️ Send the reminder text you want to save (or type 'cancel').")
    return WAITING_FOR_REMINDER

async def handle_reminder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.lower() == "cancel":
        await update.message.reply_text("Cancelled.")
        return ConversationHandler.END
    try:
        save_reminder(text)
        await update.message.reply_text("✅ Reminder saved.")
    except Exception as e:
        await update.message.reply_text(f"❌ Could not save reminder: {e}")
    return ConversationHandler.END

# ---------- /trend flow ----------
async def trend_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if args:
        t = args[0].strip().lower()
        if t not in ("bullish", "bearish", "sideways"):
            await update.message.reply_text("Please specify trend as one of: bullish, bearish, sideways.")
            return ConversationHandler.END
        save_trend(t)
        await update.message.reply_text(f"✅ Trend saved as '{t}'.")
        return ConversationHandler.END
    await update.message.reply_text("📈 Enter current market trend (bullish / bearish / sideways) — reply now.")
    return WAITING_FOR_TREND

async def handle_trend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()
    if text not in ("bullish", "bearish", "sideways"):
        await update.message.reply_text("Please reply with: bullish, bearish, or sideways.")
        return ConversationHandler.END
    try:
        save_trend(text)
        await update.message.reply_text(f"✅ Trend saved as '{text}'.")
    except Exception as e:
        await update.message.reply_text(f"❌ Could not save trend: {e}")
    return ConversationHandler.END

# ---------- /weekly ----------
async def weekly_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # compute date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    # load mistakes from last 7 days
    cutoff = start_date.isoformat()
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, text, embedding, created_at FROM mistakes WHERE created_at >= %s ORDER BY id ASC", (cutoff,))
        rows = c.fetchall()
    records = []
    for rid, text, emb_json, created_at in rows:
        try:
            vec = np.array(json.loads(emb_json), dtype=float)
            records.append({"id": rid, "text": text, "vec": vec})
        except Exception:
            continue
    groups = group_by_similarity(records, SIM_THRESHOLD)
    group_lines = top_group_lines_for_prompt(groups, limit=10)
    reminders = load_recent_reminders(days=7)
    rem_lines = reminders_lines_for_prompt(reminders)
    latest_trend = get_latest_trend()
    latest_trend_text = latest_trend["trend"] if latest_trend else "unknown"

    prompt = PROMPT_WEEKLY.format(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        group_lines=group_lines,
        latest_trend=latest_trend_text,
        reminders=rem_lines
    )
    resp = call_gemini(prompt)
    if not resp:
        resp = "(Gemini unavailable) Weekly summary could not be generated. Try again later."
    await update.message.reply_text(f"🗓️ Weekly Summary ({start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}):\n\n{resp}")

# ---------- /newses ----------
async def newses_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dt = datetime.utcnow().strftime("%Y-%m-%d")
    prompt = PROMPT_NEWSES.format(date=dt)
    resp = call_gemini(prompt)
    if not resp:
        resp = "(Gemini unavailable) News summary could not be generated. Try again later."
    await update.message.reply_text(f"📰 Top Crypto News for {dt} (Gemini-based):\n\n{resp}")

# ---------- Utility: old generate_suggestions kept as fallback ----------
PROMPT_TEMPLATE = """You are a strict but concise trading mentor.
You will receive grouped trading mistakes from my journal (already deduplicated by meaning).
Your tasks:
1) List the repeating mistakes from most frequent to least, with counts in parentheses.
2) Produce a PRE-TRADE CHECKLIST of 3–5 bullet points that specifically prevents the top repeating mistakes.
Rules:
- Be ultra concise (max ~120 words total).
- No fluff, no lecturing. Just practical reminders.
- Use plain language.

Grouped mistakes (label -> count):
{group_lines}
"""

def generate_suggestions(groups: list[dict]) -> str:
    if not groups:
        return ("No past mistakes yet — clean slate! 🚀\n\n"
                "==========Pre-trade checklist:=========\n"
                "• Confirm trend and timeframe alignment\n"
                "• Define entry, SL, TP before placing order\n"
                "• Wait for confirmation; no FOMO\n")

    trimmed = groups[:MAX_GROUPS_FOR_PROMPT]
    lines = [f"- {g['label']} -> {g['count']}" for g in trimmed]
    prompt = PROMPT_TEMPLATE.format(group_lines="\n".join(lines))

    try:
        resp = gen_model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            raise ValueError("Empty response")
        return text
    except Exception:
        top = [g["label"].lower() for g in trimmed[:3]]
        rules = []
        if any("sl" in t or "stop" in t for t in top):
            rules.append("• Set/verify stop-loss before entry")
        if any("fib" in t or "fibo" in t for t in top):
            rules.append("• Recheck Fibonacci levels & anchors")
        rules.append("• Confirm trend & structure before entry")
        rules.append("• Wait for your exact setup; no early entry")
        return ("(Gemini unavailable; showing fallback)\n" +
                "\n".join([f"{i+1}. {g['label']} ({g['count']})" for i, g in enumerate(trimmed)]) +
                "\n\nPre-trade checklist:\n" + "\n".join(rules))

# ---------- News Flash (kept) ----------
def fetch_news_brief():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    prompt = f"""
    Today is {today}.
    Give me a short 2–3 sentence news flash focused mainly on CRYPTO
    (and secondarily on global financial markets if relevant).
    At the end, add a witty/funny remark connected to the news.
    Keep it under 70 words.
    """
    try:
        resp = gen_model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception:
        return f"{today}: Crypto market quiet today... maybe even Bitcoin is napping 💤"

# ---------- Main and handlers wiring ----------
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Cancelled.")
    return ConversationHandler.END

def main():
    init_db()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CommandHandler("search", search_start),
            CommandHandler("reminder", reminder_start),
            CommandHandler("trend", trend_start),
        ],
        states={
            WAITING_FOR_MISTAKE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_mistake)],
            WAITING_FOR_SEARCH: [MessageHandler(filters.TEXT & ~filters.COMMAND,handle_search)],
            WAITING_FOR_REMINDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_reminder)],
            WAITING_FOR_TREND: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trend)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        per_user=True,
        per_chat=True,
        allow_reentry=True
    )

    # Simple handlers (non-conversation)
    app.add_handler(CommandHandler("weekly", weekly_summary))
    app.add_handler(CommandHandler("newses", newses_command))

    app.add_handler(conv_handler)

    print("🤖 Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
