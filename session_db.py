import sqlite3
import os
import logging
logging.getLogger("watchdog").setLevel(logging.ERROR)

DB_PATH = "data/session_state.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create chat_sessions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            session_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create session_history table with proper foreign key
    c.execute("""
        CREATE TABLE IF NOT EXISTS session_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            question TEXT,
            answer TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
        )
    """)
    
    # Create session_meta table
    c.execute("""
        CREATE TABLE IF NOT EXISTS session_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    # Check if session_history table has the correct structure
    c.execute("PRAGMA table_info(session_history)")
    columns = [column[1] for column in c.fetchall()]
    
    # If session_id column doesn't exist, recreate the table
    if 'session_id' not in columns:
        print("Recreating session_history table with correct schema...")
        c.execute("DROP TABLE IF EXISTS session_history")
        c.execute("""
            CREATE TABLE session_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                question TEXT,
                answer TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
            )
        """)
    
    conn.commit()
    conn.close()

def get_all_active_sessions():
    """Get all active sessions for admin view"""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT cs.session_id, cs.username, cs.session_name, cs.created_at,
                   COUNT(sh.id) as message_count,
                   MAX(sh.timestamp) as last_activity
            FROM chat_sessions cs
            LEFT JOIN session_history sh ON cs.session_id = sh.session_id
            WHERE cs.created_at > datetime('now', '-30 days')
            GROUP BY cs.session_id, cs.username, cs.session_name, cs.created_at
            ORDER BY cs.created_at DESC
        """)
        sessions = []
        for row in c.fetchall():
            sessions.append({
                "session_id": row[0],
                "username": row[1], 
                "session_name": row[2],
                "created_at": row[3],
                "message_count": row[4],
                "last_activity": row[5] or row[3]  # Use created_at if no messages
            })
        conn.close()
        return sessions
    except sqlite3.OperationalError as e:
        print(f"Database error in get_all_active_sessions: {e}")
        return []

def kill_user_session(session_id):
    """Kill a specific user session"""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Get session info before deletion
        c.execute("SELECT username, session_name FROM chat_sessions WHERE session_id = ?", (session_id,))
        session_info = c.fetchone()
        
        # Delete the session (CASCADE will handle history)
        c.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        
        return session_info
    except sqlite3.OperationalError as e:
        print(f"Database error in kill_user_session: {e}")
        return None

def create_new_session(username, session_name="New Chat"):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chat_sessions (username, session_name) VALUES (?, ?)", (username, session_name))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id

def rename_session(session_id, new_name):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE chat_sessions SET session_name = ? WHERE session_id = ?", (new_name, session_id))
    conn.commit()
    conn.close()

def delete_session(session_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Delete history first (though CASCADE should handle this)
    c.execute("DELETE FROM session_history WHERE session_id = ?", (session_id,))
    # Delete session
    c.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def get_user_sessions(username):
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT session_id, session_name, created_at FROM chat_sessions WHERE username = ? ORDER BY created_at DESC", (username,))
        sessions = [{"session_id": row[0], "session_name": row[1], "created_at": row[2]} for row in c.fetchall()]
        conn.close()
        return sessions
    except sqlite3.OperationalError as e:
        print(f"Database error in get_user_sessions: {e}")
        return []

def save_qa_history(session_id, history):
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Clear existing history for this session
        c.execute("DELETE FROM session_history WHERE session_id = ?", (session_id,))
        
        # Insert new history
        for qa in history:
            c.execute("INSERT INTO session_history (session_id, question, answer) VALUES (?, ?, ?)", 
                     (session_id, qa["question"], qa["answer"]))
        
        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"Database error in save_qa_history: {e}")
        # Reinitialize database and try again
        init_db()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM session_history WHERE session_id = ?", (session_id,))
        for qa in history:
            c.execute("INSERT INTO session_history (session_id, question, answer) VALUES (?, ?, ?)", 
                     (session_id, qa["question"], qa["answer"]))
        conn.commit()
        conn.close()

def load_qa_history(session_id):
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT question, answer, timestamp FROM session_history WHERE session_id = ? ORDER BY timestamp", (session_id,))
        rows = c.fetchall()
        conn.close()
        return [{"question": q, "answer": a, "timestamp": t} for q, a, t in rows]
    except sqlite3.OperationalError as e:
        print(f"Database error in load_qa_history: {e}")
        return []

def save_meta(key, value):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("REPLACE INTO session_meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def load_meta(key, default=None):
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT value FROM session_meta WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else default
    except sqlite3.OperationalError:
        return default
    
def get_active_sessions(username):
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Assume sessions expire after 30 days of inactivity
        c.execute("""
            SELECT session_id, session_name, created_at 
            FROM chat_sessions 
            WHERE username = ? AND created_at > datetime('now', '-30 days')
            ORDER BY created_at DESC
        """, (username,))
        sessions = [{"session_id": row[0], "session_name": row[1], "created_at": row[2]} for row in c.fetchall()]
        conn.close()
        return sessions
    except sqlite3.OperationalError as e:
        print(f"Database error in get_active_sessions: {e}")
        return []
    
def cleanup_inactive_sessions():
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM chat_sessions WHERE created_at < datetime('now', '-30 days')")
        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"Database error in cleanup_inactive_sessions: {e}")

def validate_admin(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user and user[0] == "admin"

# Force database initialization on import
init_db()