import sqlite3
from passlib.hash import bcrypt
import os
import logging
logging.getLogger("watchdog").setLevel(logging.ERROR)

DB_PATH = "data/auth.db"

def init_auth_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            hashed_password TEXT NOT NULL,
            role TEXT NOT NULL,
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        from passlib.hash import bcrypt
        default_admin_user = "admin"
        default_admin_pass = "admin123"  # Change after first login!
        hashed = bcrypt.hash(default_admin_pass)
        c.execute(
            "INSERT INTO users (username, hashed_password, role, created_by) VALUES (?, ?, ?, ?)",
            (default_admin_user, hashed, "admin", "system")
        )
        print(f"Default admin created: username='{default_admin_user}', password='{default_admin_pass}'")
    conn.commit()
    conn.close()

def add_user(username, password, role, created_by):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hashed = bcrypt.hash(password)
    c.execute("INSERT INTO users (username, hashed_password, role, created_by) VALUES (?, ?, ?, ?)",
              (username, hashed, role, created_by))
    conn.commit()
    conn.close()

def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, hashed_password, role FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"username": row[0], "hashed_password": row[1], "role": row[2]}
    return None

def validate_user(username, password):
    user = get_user(username)
    if user and bcrypt.verify(password, user["hashed_password"]):
        return user
    return None

def list_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, role, created_by, created_at FROM users")
    users = c.fetchall()
    conn.close()
    return users