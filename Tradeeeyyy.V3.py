import os
import json
from datetime import datetime
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler

# --- Gemini SDK ---
import google.generativeai as genai

# --- PostgreSQL ---
import psycopg2

# ========= CONFIG =========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyANRoUqjChN8Ygtp_pCguwGLfXIhRgHdQ0")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8131796331:AAEkI-rs8-WyD9MAfl0cG_eZFrywe5utIFo")

EMBEDDING_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-2.5-flash"
SIM_THRESHOLD = 0.78
MAX_GROUPS_FOR_PROMPT = 20

# States for conversation
WAITING_FOR_MISTAKE = 1
# ==========================

# ---------- Gemini setup ----------
genai.configure(api_key=GEMINI_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)

# ---------- DB helpers (PostgreSQL version) ----------
def db_conn():
    """
    Connect to Postgres using Render DATABASE_URL.
    """
    return psycopg2.connect(os.environ["DATABASE_URL"], sslmode="require")

def init_db():
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS mistakes (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

def save_mistake(text: str, embedding: list[float]):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO mistakes (text, embedding, created_at) VALUES (%s, %s, %s)",
            (json.dumps(text) if isinstance(text, list) else text, json.dumps(embedding), datetime.utcnow().isoformat())
        )
        conn.commit()

def load_mistakes():
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, text, embedding FROM mistakes ORDER BY id ASC")
        rows = c.fetchall()
    records = []
    for rid, text, emb_json in rows:
        try:
            vec = np.array(json.loads(emb_json), dtype=float)
            records.append({"id": rid, "text": text, "vec": vec})
        except Exception:
            continue
    return records

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

# ---------- Gemini generation ----------
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

# ---------- News Flash ----------
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

# ---------- Telegram Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    init_db()

    # News
    news = fetch_news_brief()
    await update.message.reply_text("📢 NEWS FLASH + FUNNY NOTE\n" + "="*40 + f"\n{news}")

    # Checklist
    records = load_mistakes()
    groups = group_by_similarity(records, SIM_THRESHOLD)
    suggestions = generate_suggestions(groups)
    await update.message.reply_text("📋 PRE-TRADE CHECKLIST\n" + "="*40 + f"\n{suggestions}")

    # Ask for mistake
    await update.message.reply_text("\nHave you made any new mistakes? Reply here (or type 'skip').")
    return WAITING_FOR_MISTAKE

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

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Cancelled.")
    return ConversationHandler.END

# ---------- Main ----------
def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            WAITING_FOR_MISTAKE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_mistake)]
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        per_user=True,
        per_chat=True,
        allow_reentry=True
    )

    app.add_handler(conv_handler)
    print("🤖 Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()