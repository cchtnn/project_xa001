import sqlite3
import os
import logging
logging.getLogger("watchdog").setLevel(logging.ERROR)

DB_PATH = "data/session_state.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS session_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS session_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_qa_history(history):
    # Initialize DB if it doesn't exist
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM session_history")
    for qa in history:
        c.execute("INSERT INTO session_history (question, answer) VALUES (?, ?)", (qa["question"], qa["answer"]))
    conn.commit()
    conn.close()

def load_qa_history():
    # Initialize DB if it doesn't exist
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT question, answer FROM session_history")
        rows = c.fetchall()
        conn.close()
        return [{"question": q, "answer": a} for q, a in rows]
    except sqlite3.OperationalError:
        # If table doesn't exist, return empty list
        return []

def save_meta(key, value):
    # Initialize DB if it doesn't exist
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("REPLACE INTO session_meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def load_meta(key, default=None):
    # Initialize DB if it doesn't exist
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT value FROM session_meta WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else default
    except sqlite3.OperationalError:
        # If table doesn't exist, return default
        return default