import sqlite3
import os

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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM session_history")
    for qa in history:
        c.execute("INSERT INTO session_history (question, answer) VALUES (?, ?)", (qa["question"], qa["answer"]))
    conn.commit()
    conn.close()

def load_qa_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT question, answer FROM session_history")
    rows = c.fetchall()
    conn.close()
    return [{"question": q, "answer": a} for q, a in rows]

def save_meta(key, value):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("REPLACE INTO session_meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def load_meta(key, default=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM session_meta WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else default