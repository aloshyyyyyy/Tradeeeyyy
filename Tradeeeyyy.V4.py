# tradie.v4.py - Part A (copy first)
import os
import json
import time
from datetime import datetime, timedelta
import numpy as np
import requests
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
DATABASE_URL = os.getenv("DATABASE_URL", None)
CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_KEY", None)  # optional
COINGECKO_BASE = os.getenv("COINGECKO_BASE", "https://api.coingecko.com/api/v3")

# Safety/timeouts
NEWS_CACHE_TTL_SECONDS = int(os.getenv("NEWS_CACHE_TTL_SECONDS", "21600"))  # default 6 hours

EMBEDDING_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-2.5-flash"
SIM_THRESHOLD = 0.78
MAX_GROUPS_FOR_PROMPT = 20

# Conversation states
WAITING_FOR_MISTAKE = 1
WAITING_FOR_SEARCH = 2
WAITING_FOR_REMINDER = 3
WAITING_FOR_TREND = 4
# Trade flow states
TRADE_COIN = 10
TRADE_ENTRY = 11
TRADE_TP = 12
TRADE_SL = 13
TRADE_SIZE = 14
TRADE_NOTES = 15
TRADE_CONFIRM = 16
# Close trade
TRADE_SELECT_CLOSE = 20
TRADE_CLOSE_RESULT = 21
TRADE_CLOSE_PNL = 22
TRADE_CLOSE_NOTES = 23

# -----------------------------
# Gemini setup
# -----------------------------
genai.configure(api_key=GEMINI_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)

# -----------------------------
# DB helpers
# -----------------------------
def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL env var not set")
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    with db_conn() as conn:
        c = conn.cursor()
        # mistakes
        c.execute("""
            CREATE TABLE IF NOT EXISTS mistakes (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """)
        # market_trends
        c.execute("""
            CREATE TABLE IF NOT EXISTS market_trends (
                id SERIAL PRIMARY KEY,
                trend TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """)
        # reminders
        c.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """)
        # trades table (new)
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                coin TEXT NOT NULL,
                entry_price DOUBLE PRECISION,
                take_profit DOUBLE PRECISION,
                stop_loss DOUBLE PRECISION,
                position_size DOUBLE PRECISION,
                notes TEXT,
                status TEXT DEFAULT 'OPEN',
                outcome TEXT,
                pnl DOUBLE PRECISION,
                created_at TIMESTAMP NOT NULL,
                closed_at TIMESTAMP
            )
        """)
        conn.commit()

# Mistakes
def save_mistake(text: str, embedding: list[float]):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO mistakes (text, embedding, created_at) VALUES (%s, %s, %s)",
            (text, json.dumps(embedding), datetime.utcnow())
        )
        conn.commit()

def load_mistakes(since_days: int = 3650):
    cutoff = (datetime.utcnow() - timedelta(days=since_days)).isoformat()
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, text, embedding, created_at FROM mistakes WHERE created_at >= %s ORDER BY id ASC", (cutoff,))
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
            (text, datetime.utcnow())
        )
        conn.commit()

def load_recent_reminders(days: int = 7):
    cutoff = datetime.utcnow() - timedelta(days=days)
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, text, created_at FROM reminders WHERE created_at >= %s ORDER BY id DESC", (cutoff,))
        rows = c.fetchall()
    return [{"id": r[0], "text": r[1], "created_at": r[2]} for r in rows]

# Trends
def save_trend(trend_text: str):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO market_trends (trend, created_at) VALUES (%s, %s)",
            (trend_text.lower(), datetime.utcnow())
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

# Trades
def create_trade(coin, entry_price, tp, sl, size, notes):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO trades (coin, entry_price, take_profit, stop_loss, position_size, notes, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id
        """, (coin, entry_price, tp, sl, size, notes, datetime.utcnow()))
        tid = c.fetchone()[0]
        conn.commit()
    return tid

def get_open_trades(limit=10):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, coin, entry_price, take_profit, stop_loss, position_size, notes, created_at FROM trades WHERE status='OPEN' ORDER BY id DESC LIMIT %s", (limit,))
        rows = c.fetchall()
    return [{"id": r[0], "coin": r[1], "entry": r[2], "tp": r[3], "sl": r[4], "size": r[5], "notes": r[6], "created_at": r[7]} for r in rows]

def get_trade_by_id(tid):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, coin, entry_price, take_profit, stop_loss, position_size, notes, status, outcome, pnl, created_at, closed_at FROM trades WHERE id=%s", (tid,))
        row = c.fetchone()
    if not row:
        return None
    keys = ["id","coin","entry_price","take_profit","stop_loss","position_size","notes","status","outcome","pnl","created_at","closed_at"]
    return dict(zip(keys, row))

def close_trade(tid, outcome, pnl, notes=None):
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("UPDATE trades SET status='CLOSED', outcome=%s, pnl=%s, closed_at=%s, notes = COALESCE(notes || E'\\nCLOSE NOTES: ' || %s, %s) WHERE id=%s", (outcome, pnl, datetime.utcnow(), notes, notes, tid))
        conn.commit()

# -----------------------------
# Embeddings & similarity
# -----------------------------
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
        if r["id"] in used: continue
        seed = r
        group_members = [seed["text"]]
        group_ids = [seed["id"]]
        used.add(seed["id"])
        for j in range(i + 1, len(records)):
            s = records[j]
            if s["id"] in used: continue
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

# -----------------------------
# Gemini prompts (same as v3.5)
# -----------------------------
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
# tradie.v4.py - Part B (copy second)
PROMPT_NEWSES = """You are a crypto news assistant. Today is {date}.

Task:
1. Consider top crypto/finance news sources.
2. Pick the 5 most important crypto news items published on {date}.
3. For each item, output:
   - Date (use {date})
   - Headline
   - ~80-100 word summary focused on trading/market implications.
   - If available, include source URL.

Rules:
- Exactly 5 items only.
- Each summary ~80-100 words.
- Order from most important → least.
- Keep language factual and trader-oriented.
- If you cannot access live web information, be explicit: start with "I cannot access live web sources; based on knowledge I have, ..." and provide best-effort summaries.
"""

# -----------------------------
# Gemini wrapper & router
# -----------------------------
def call_gemini(prompt: str):
    try:
        resp = gen_model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text
    except Exception:
        return None

def parse_intent_with_gemini(user_text: str):
    """
    Ask Gemini to parse user text into an intent JSON.
    The prompt instructs Gemini to output strict JSON.
    """
    prompt = f"""
You are an intent parser for a crypto trading assistant.
User input: \"\"\"{user_text}\"\"\"

Return a JSON object only (no extra text) with fields:
- intent: one of ["greeting", "price", "top", "news", "trade_start", "trade_close", "mistake", "reminder", "trend", "weekly", "search", "unknown"]
- entities: object with keys depending on intent, e.g. {{ "coin": "bitcoin", "coins": ["btc","eth"], "type": "gainers" }}
- reply: optional short user-facing hint if needed.

Examples:
Input: "what's the price of bitcoin and eth?"
Output: {{"intent":"price","entities":{{"coins":["bitcoin","eth"]}}}}

Input: "gm, i'm opening a trade"
Output: {{"intent":"trade_start","entities":{{}}}}

Now parse and output JSON only.
"""
    out = call_gemini(prompt)
    if not out:
        return {"intent":"unknown","entities":{}}
    # try to extract JSON from out
    try:
        # some models might include code fences; try to find first { ... }
        first = out.find("{")
        last = out.rfind("}")
        if first != -1 and last != -1:
            jtxt = out[first:last+1]
            return json.loads(jtxt)
        return {"intent":"unknown","entities":{}}
    except Exception:
        return {"intent":"unknown","entities":{}}

# -----------------------------
# CoinGecko helpers
# -----------------------------
def coingecko_simple_price(coins: list, vs_currency: str = "usd"):
    ids = ",".join(coins)
    url = f"{COINGECKO_BASE}/simple/price?ids={ids}&vs_currencies={vs_currency}&include_24hr_change=true&include_24hr_vol=true"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def coingecko_markets(vs_currency="usd", per_page=10, page=1):
    url = f"{COINGECKO_BASE}/coins/markets?vs_currency={vs_currency}&order=market_cap_desc&per_page={per_page}&page={page}&price_change_percentage=24h"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# -----------------------------
# CryptoPanic helpers (news) + caching
# -----------------------------
_NEWS_CACHE = {"ts": 0, "items": None}

def fetch_cryptopanic_top(limit=10):
    """
    Return list of news items (dicts) from CryptoPanic.
    """
    # use cache ttl
    now = time.time()
    if _NEWS_CACHE["items"] and (now - _NEWS_CACHE["ts"] < NEWS_CACHE_TTL_SECONDS):
        return _NEWS_CACHE["items"]
    if not CRYPTOPANIC_KEY:
        return None
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_KEY}&public=true&kind=news&regions=us,global&filter=important"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        posts = data.get("results") or data.get("results") or []
        # cache top limit
        items = posts[:limit]
        _NEWS_CACHE["items"] = items
        _NEWS_CACHE["ts"] = now
        return items
    except Exception:
        return None

# -----------------------------
# Helpers and prompt builders
# -----------------------------
def top_group_lines_for_prompt(groups: list[dict], limit: int = 20):
    trimmed = groups[:limit]
    lines = [f"- {g['label']} -> {g['count']}" for g in trimmed]
    return "\n".join(lines)

def reminders_lines_for_prompt(reminders_list):
    if not reminders_list:
        return ""
    return "\n".join([f"• {r['text']}" for r in reminders_list])

# -----------------------------
# Handlers
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    init_db()
    # Greeting
    await update.message.reply_text("👋 Good to see you. Running quick check...")
    # Fetch brief news + checklist
    news_brief = None
    try:
        news_brief = fetch_news_brief()  # existing short flash from v3.5
    except Exception:
        news_brief = None
    if news_brief:
        await update.message.reply_text("📢 NEWS FLASH\n" + news_brief)
    # checklist (trend aware)
    records = load_mistakes()
    groups = group_by_similarity(records, SIM_THRESHOLD)
    latest_trend = get_latest_trend()
    latest_trend_text = latest_trend["trend"] if latest_trend else "unknown"
    reminders = load_recent_reminders(days=30)
    group_lines = top_group_lines_for_prompt(groups)
    rem_lines = reminders_lines_for_prompt(reminders)
    prompt = PROMPT_CHECKLIST.format(
        group_lines=group_lines,
        reminders=rem_lines,
        latest_trend=latest_trend_text,
        date=datetime.utcnow().strftime("%Y-%m-%d")
    )
    suggestions = call_gemini(prompt) or generate_suggestions(groups)
    await update.message.reply_text("📋 PRE-TRADE CHECKLIST\n" + suggestions)
    await update.message.reply_text("\nHave you made any new mistakes? Reply (or type 'skip').")
    return WAITING_FOR_MISTAKE

# Mistake flow (kept)
async def handle_mistake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.lower() == "skip":
        await update.message.reply_text("👍 Skipped.")
        return ConversationHandler.END
    try:
        emb = embed_text(text)
        save_mistake(text, emb)
        await update.message.reply_text("✅ Saved!")
    except Exception as e:
        await update.message.reply_text(f"❌ Could not save mistake: {e}")
    return ConversationHandler.END

# -----------------------------
# Search flow (kept)
# -----------------------------
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

# -----------------------------
# Reminder flow (kept)
# -----------------------------
async def reminder_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

# -----------------------------
# Trend flow (kept)
# -----------------------------
async def trend_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if args:
        t = args[0].strip().lower()
        if t not in ("bullish", "bearish", "sideways"):
            await update.message.reply_text("Please specify trend: bullish, bearish, sideways.")
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

# -----------------------------
# Weekly summary (expanded to include trades)
# -----------------------------
async def weekly_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    # mistakes in last 7 days
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
    # trades summary in last 7 days
    with db_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT outcome, pnl FROM trades WHERE created_at >= %s OR (closed_at IS NOT NULL AND closed_at >= %s)", (cutoff, cutoff))
        trows = c.fetchall()
    wins = sum(1 for r in trows if r[0] and r[0].lower()=="win")
    losses = sum(1 for r in trows if r[0] and r[0].lower()=="loss")
    total_trades = len(trows)
    avg_pnl = None
    if trows:
        pnls = [r[1] for r in trows if r[1] is not None]
        if pnls:
            avg_pnl = sum(pnls)/len(pnls)
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
    # Append quick trade stats
    stats = f"\n\nQuick trade stats (last 7d): total={total_trades}, wins={wins}, losses={losses}, avg_pnl={avg_pnl if avg_pnl is not None else 'N/A'}"
    await update.message.reply_text(f"🗓️ Weekly Summary ({start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}):\n\n{resp}{stats}")

# -----------------------------
# News command (Hybrid: CryptoPanic + Gemini summarization)
# -----------------------------
async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dt = datetime.utcnow().strftime("%Y-%m-%d")
    # Try CryptoPanic first (cached)
    items = fetch_cryptopanic_top(limit=10)
    if items:
        # Build news_list text for Gemini
        news_list = []
        for p in items[:5]:
            title = p.get("title")
            url = p.get("url")
            published = p.get("published_at", dt)
            # try to keep short content for gemini
            news_list.append(f"{published[:10]} - {title} ({url})")
        # pass to Gemini for polish & 80-100 word summaries each
        prompt = f"Today is {dt}.\nSummarize these 5 crypto news items in 80-100 words each, with date and source URL:\n\n" + "\n".join(news_list)
        resp = call_gemini(prompt)
        if resp:
            # chunk if too long
            CHUNK = 3500
            text = resp.strip()
            for i in range(0, len(text), CHUNK):
                await update.message.reply_text(text[i:i+CHUNK])
            return
    # Fallback: call Gemini-only news prompt
    prompt = PROMPT_NEWSES.format(date=dt)
    resp = call_gemini(prompt)
    if not resp:
        resp = "(Gemini unavailable) News summary could not be generated. Try again later."
    # chunk for Telegram
    CHUNK = 3500
    text = resp.strip()
    for i in range(0, len(text), CHUNK):
        await update.message.reply_text(text[i:i+CHUNK])

# -----------------------------
# Market command (CoinGecko)
# -----------------------------
async def market_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if args:
        # map symbols to ids naively (user may pass 'btc' or 'bitcoin')
        coins = [a.strip().lower() for a in args]
        # try to use CoinGecko ids: user might pass symbol; for quick mapping we assume common ids
        # ideally you'd maintain a mapping; here we do a quick heuristic: pass coins join (CoinGecko accepts ids like 'bitcoin,ethereum')
        ids = []
        for c in coins:
            # common cases
            lookup = {"btc":"bitcoin","eth":"ethereum","bnb":"binancecoin","xrp":"ripple","ada":"cardano","sol":"solana"}
            ids.append(lookup.get(c,c))
        data = coingecko_simple_price(ids, vs_currency="usd")
        if not data:
            await update.message.reply_text("❌ Could not fetch prices from CoinGecko.")
            return
        lines = []
        for k, v in data.items():
            price = v.get("usd")
            change = v.get("usd_24h_change")
            vol = v.get("usd_24h_vol")
            lines.append(f"{k.capitalize()}: ${price:.4f} (24h: {change:.2f}%), vol: {vol:.0f}")
        await update.message.reply_text("\n".join(lines))
        return
    # default top coins
    markets = coingecko_markets(per_page=6)
    if not markets:
        await update.message.reply_text("❌ Could not fetch market data.")
        return
    lines = []
    for m in markets:
        name = m.get("name")
        symbol = m.get("symbol").upper()
        price = m.get("current_price")
        ch = m.get("price_change_percentage_24h")
        vol = m.get("total_volume")
        lines.append(f"{name} ({symbol}): ${price:.2f} ({ch:.2f}%) vol {int(vol):,}")
    await update.message.reply_text("Top coins:\n" + "\n".join(lines))

# -----------------------------
# Top movers command
# -----------------------------
async def top_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    tp = "gainers"
    if args and args[0].lower() in ("losers","gainers"):
        tp = args[0].lower()
    markets = coingecko_markets(per_page=50)
    if not markets:
        await update.message.reply_text("❌ Could not fetch market data.")
        return
    # sort
    markets_sorted = sorted(markets, key=lambda m: m.get("price_change_percentage_24h") or 0, reverse=(tp=="gainers"))
    top5 = markets_sorted[:5]
    lines = []
    for m in top5:
        lines.append(f"{m.get('name')} ({m.get('symbol').upper()}): {m.get('price_change_percentage_24h'):.2f}% | ${m.get('current_price'):.2f}")
    await update.message.reply_text(f"Top {tp}:\n" + "\n".join(lines))

# -----------------------------
# Trade flows
# -----------------------------
async def trade_start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🟢 Starting trade creation. Which coin/pair are you trading? (e.g. bitcoin or BTC/USDT)")
    return TRADE_COIN

async def trade_collect_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    context.user_data['trade'] = {'coin': text}
    await update.message.reply_text("Entry price?")
    return TRADE_ENTRY

async def trade_collect_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    try:
        entry = float(txt)
    except Exception:
        await update.message.reply_text("Please send a numeric entry price.")
        return TRADE_ENTRY
    context.user_data['trade']['entry'] = entry
    await update.message.reply_text("Take profit (or 'none')?")
    return TRADE_TP

async def trade_collect_tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    tp = None
    if txt.lower() != "none":
        try:
            tp = float(txt)
        except Exception:
            await update.message.reply_text("Please send numeric TP or 'none'.")
            return TRADE_TP
    context.user_data['trade']['tp'] = tp
    await update.message.reply_text("Stop loss (or 'none')?")
    return TRADE_SL

async def trade_collect_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    sl = None
    if txt.lower() != "none":
        try:
            sl = float(txt)
        except Exception:
            await update.message.reply_text("Please send numeric SL or 'none'.")
            return TRADE_SL
    context.user_data['trade']['sl'] = sl
    await update.message.reply_text("Position size (units or USD amount)? (or 'none')")
    return TRADE_SIZE

async def trade_collect_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    size = None
    if txt.lower() != "none":
        try:
            size = float(txt)
        except Exception:
            await update.message.reply_text("Please send numeric position size or 'none'.")
            return TRADE_SIZE
    context.user_data['trade']['size'] = size
    await update.message.reply_text("Any notes? (or 'skip')")
    return TRADE_NOTES

async def trade_collect_notes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    if txt.lower() == "skip":
        txt = None
    context.user_data['trade']['notes'] = txt
    # confirm
    tr = context.user_data['trade']
    msg = f"Please confirm trade:\nCoin: {tr.get('coin')}\nEntry: {tr.get('entry')}\nTP: {tr.get('tp')}\nSL: {tr.get('sl')}\nSize: {tr.get('size')}\nNotes: {tr.get('notes')}\n\nType 'yes' to confirm or 'cancel'."
    await update.message.reply_text(msg)
    return TRADE_CONFIRM

async def trade_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip().lower()
    if txt == "cancel":
        await update.message.reply_text("Trade creation cancelled.")
        return ConversationHandler.END
    if txt != "yes":
        await update.message.reply_text("Please type 'yes' or 'cancel'.")
        return TRADE_CONFIRM
    tr = context.user_data.get('trade', {})
    tid = create_trade(tr.get('coin'), tr.get('entry'), tr.get('tp'), tr.get('sl'), tr.get('size'), tr.get('notes'))
    await update.message.reply_text(f"✅ Trade saved with id {tid}. Use /trade_close when you close it.")
    return ConversationHandler.END

# Close trade flows
async def trade_close_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    opens = get_open_trades(limit=10)
    if not opens:
        await update.message.reply_text("No open trades found.")
        return ConversationHandler.END
    context.user_data['open_trades'] = opens
    lines = [f"{t['id']}: {t['coin']} entry {t['entry']} tp {t['tp']} sl {t['sl']}" for t in opens]
    await update.message.reply_text("Open trades:\n" + "\n".join(lines) + "\nReply with trade id to close.")
    return TRADE_SELECT_CLOSE

async def trade_close_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    try:
        tid = int(txt)
    except Exception:
        await update.message.reply_text("Send a valid trade id.")
        return TRADE_SELECT_CLOSE
    tr = get_trade_by_id(tid)
    if not tr:
        await update.message.reply_text("Trade not found.")
        return ConversationHandler.END
    context.user_data['closing_trade'] = tr
    await update.message.reply_text("Outcome? (win/loss)")
    return TRADE_CLOSE_RESULT

async def trade_close_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip().lower()
    if txt not in ("win","loss","lossy","lose"):
        await update.message.reply_text("Please reply with 'win' or 'loss'.")
        return TRADE_CLOSE_RESULT
    outcome = "WIN" if txt=="win" else "LOSS"
    context.user_data['closing_trade']['outcome'] = outcome
    await update.message.reply_text("PnL amount (e.g. -12.5 or 10.2) (send number)")
    return TRADE_CLOSE_PNL

async def trade_close_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    try:
        pnl = float(txt)
    except Exception:
        await update.message.reply_text("Send numeric PnL amount (negative for loss).")
        return TRADE_CLOSE_PNL
    context.user_data['closing_trade']['pnl'] = pnl
    await update.message.reply_text("Any closing notes? (or 'skip')")
    return TRADE_CLOSE_NOTES

async def trade_close_notes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip()
    if txt.lower() == "skip":
        txt = None
    tr = context.user_data['closing_trade']
    tid = tr['id']
    outcome = tr['outcome']
    pnl = tr['pnl']
    close_trade(tid, outcome, pnl, txt)
    # If loss -> ask to save mistake
    if outcome == "LOSS":
        await update.message.reply_text("Do you want to save a mistake note for this loss? (type it now or 'no')")
        # store state: after message, if not 'no' save as mistake
        context.user_data['awaiting_loss_mistake_for'] = tid
        return WAITING_FOR_MISTAKE
    else:
        await update.message.reply_text(f"✅ Trade {tid} closed as {outcome} with PnL {pnl}. Good job!")
        return ConversationHandler.END

# -----------------------------
# Helper: existing fetch_news_brief (kept)
# -----------------------------
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
        resp = call_gemini(prompt)
        return (resp or "").strip()
    except Exception:
        return f"{today}: Crypto market quiet today... maybe even Bitcoin is napping 💤"

# -----------------------------
# Free-text router (single entry point)
# -----------------------------
async def free_text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    # quick guard for commands
    if text.startswith("/"):
        return
    # parse intent with Gemini
    parsed = parse_intent_with_gemini(text)
    intent = parsed.get("intent", "unknown")
    entities = parsed.get("entities", {})
    # handle intents
    if intent == "greeting":
        await update.message.reply_text("Hey! Do you want a quick market check or planning a trade?")
        return
    if intent == "price":
        coins = entities.get("coins") or entities.get("coin")
        if not coins:
            await update.message.reply_text("Which coin(s) do you want prices for? (e.g. BTC, ETH)")
            return
        # normalize to list
        if isinstance(coins, str):
            coins = [coins]
        await handle_price_intent(update, context, coins)
        return
    if intent == "top":
        kind = entities.get("type", "gainers")
        # reuse top_command logic
        context.args = [kind]
        await top_command(update, context)
        return
    if intent == "news":
        await news_command(update, context)
        return
    if intent == "trade_start":
        await trade_start_cmd(update, context)
        return
    if intent == "trade_close":
        await trade_close_cmd(update, context)
        return
    if intent == "mistake":
        await update.message.reply_text("Okay — describe the mistake you want to save.")
        return WAITING_FOR_MISTAKE
    if intent == "reminder":
        await update.message.reply_text("Send the reminder text to save.")
        return WAITING_FOR_REMINDER
    if intent == "trend":
        await update.message.reply_text("Send trend (bullish / bearish / sideways).")
        return WAITING_FOR_TREND
    if intent == "weekly":
        await weekly_summary(update, context)
        return
    if intent == "search":
        await search_start(update, context)
        return
    # unknown
    await update.message.reply_text("Sorry, I couldn't understand. Try a direct command like /market or say 'price of BTC'.")

# -----------------------------
# Helper to handle price intent quickly
# -----------------------------
async def handle_price_intent(update: Update, context: ContextTypes.DEFAULT_TYPE, coins):
    # map common symbols
    lookup = {"btc":"bitcoin","eth":"ethereum","bnb":"binancecoin","xrp":"ripple","ada":"cardano","sol":"solana","doge":"dogecoin"}
    ids = []
    for c in coins:
        k = c.strip().lower()
        ids.append(lookup.get(k, k))
    data = coingecko_simple_price(ids, vs_currency="usd")
    if not data:
        await update.message.reply_text("❌ Could not fetch prices.")
        return
    lines = []
    for k, v in data.items():
        price = v.get("usd")
        ch = v.get("usd_24h_change")
        lines.append(f"{k.capitalize()}: ${price:.4f} (24h {ch:.2f}%)")
    # brief Gemini commentary
    comment_prompt = f"Given these prices:\n{json.dumps(data)}\nGive a 2-sentence trader-focused commentary."
    comment = call_gemini(comment_prompt) or ""
    await update.message.reply_text("\n".join(lines) + ("\n\n" + comment if comment else ""))

# -----------------------------
# Main wiring
# -----------------------------
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
            CommandHandler("trade_start", trade_start_cmd),
            CommandHandler("trade_close", trade_close_cmd),
        ],
        states={
            WAITING_FOR_MISTAKE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_mistake)],
            WAITING_FOR_SEARCH: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_search)],
            WAITING_FOR_REMINDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_reminder)],
            WAITING_FOR_TREND: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_trend)],
            TRADE_COIN: [MessageHandler(filters.TEXT & ~filters.COMMAND, trade_collect_coin)],
            TRADE_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, trade_collect_entry)],
            TRADE_TP: [MessageHandler(filters.TEXT & ~filters.COMMAND, trade_collect_tp)],
            TRADE_SL: [MessageHandler(filters.TEXT & ~filters.COMMAND, trade_collect_sl)],
            TRADE_SIZE: [MessageHandler(filters.TEXT & ~filters.COMMAND, trade_collect_size)],
            TRADE_NOTES: [MessageHandler(filters.TEXT & ~filters.COMMAND, trade_collect_notes)],
            TRADE_CONFIRM: [MessageHandler(filters.TEXT & ~filters.COMMAND, trade_confirm)],
            TRADE_SELECT_CLOSE: [MessageHandler(filters.TEXT & ~filters.COMMAND, trade_close_select)],
            TRADE_CLOSE_RESULT: [MessageHandler(filters.TEXT & ~filters.COMMAND, trade_close_result)],
            TRADE_CLOSE_PNL: [MessageHandler(filters.TEXT & ~filters.COMMAND, trade_close_pnl)],
            TRADE_CLOSE_NOTES: [MessageHandler(filters.TEXT & ~filters.COMMAND, trade_close_notes)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        per_user=True,
        per_chat=True,
        allow_reentry=True
    )

    # simple handlers
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("weekly", weekly_summary))
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("newses", news_command))
    app.add_handler(CommandHandler("market", market_command))
    app.add_handler(CommandHandler("top", top_command))
    app.add_handler(CommandHandler("trade", trade_start_cmd))  # alias
    app.add_handler(CommandHandler("trade_close", trade_close_cmd))
    app.add_handler(CommandHandler("mytrades", lambda u,c: u.message.reply_text("Use /mytrades to get list (not implemented)")))
    # free-text router (last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, free_text_router))

    print("🤖 Tradie V4 running...")
    app.run_polling()

if __name__ == "__main__":
    main()
