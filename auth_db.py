import sqlite3
from passlib.hash import bcrypt
import os
import logging
import re

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
        default_admin_user = "admin"
        default_admin_pass = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")
        if default_admin_pass == "admin123":
            logging.warning("Using default admin password 'admin123'. Change it immediately after first login!")
        hashed = bcrypt.hash(default_admin_pass)
        c.execute(
            "INSERT INTO users (username, hashed_password, role, created_by) VALUES (?, ?, ?, ?)",
            (default_admin_user, hashed, "admin", "system")
        )
        logging.info(f"Default admin created: username='{default_admin_user}'")
    conn.commit()
    conn.close()

def add_user(username, password, role, created_by, bypass_password_validation=False):
    if not bypass_password_validation:
        if not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{12,}$", password):
            raise ValueError("Password must be at least 12 characters, with uppercase, lowercase, numbers, and special characters")
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

def validate_admin(username):
    user = get_user(username)
    return user and user["role"] == "admin"

def list_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, role, created_by, created_at FROM users")
    users = c.fetchall()
    conn.close()
    return users

def update_user(original_username, new_username, new_password, new_role):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if new_password:
        if not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{12,}$", new_password):
            raise ValueError("Password must be at least 12 characters, with uppercase, lowercase, numbers, and special characters")
        hashed = bcrypt.hash(new_password)
        c.execute("UPDATE users SET username=?, hashed_password=?, role=? WHERE username=?",
                  (new_username, hashed, new_role, original_username))
    else:
        c.execute("UPDATE users SET username=?, role=? WHERE username=?",
                  (new_username, new_role, original_username))
    conn.commit()
    conn.close()

def delete_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()