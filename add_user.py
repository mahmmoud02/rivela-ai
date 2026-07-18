"""
Add a user to Rivela AI.

Usage:
    python add_user.py <username> <password> [role]

Role defaults to "doctor". Examples:
    python add_user.py drsmith pass123
    python add_user.py drsmith pass123 doctor
    python add_user.py admin1  secret  admin
"""

import sys
import sqlite3
import os
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), "instance", "chestai.db")

def add_user(username, password, role="doctor"):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), role),
        )
        conn.commit()
        print(f"OK User '{username}' added with role '{role}'.")
    except sqlite3.IntegrityError:
        print(f"ERROR Username '{username}' already exists.")
    finally:
        conn.close()

def list_users():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, username, role, created_at FROM users ORDER BY id").fetchall()
    conn.close()
    if not rows:
        print("No users found.")
        return
    print(f"{'ID':<4} {'Username':<20} {'Role':<10} {'Created'}")
    print("-" * 55)
    for r in rows:
        print(f"{r[0]:<4} {r[1]:<20} {r[2]:<10} {r[3][:16]}")

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "list":
        list_users()
    elif len(args) >= 2:
        username = args[0]
        password = args[1]
        role     = args[2] if len(args) >= 3 else "doctor"
        add_user(username, password, role)
        print()
        list_users()
    else:
        print(__doc__)
