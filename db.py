import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "instance", "chestai.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")


#to add database
#  python -c "from werkzeug.security import generate_password_hash; import sqlite3; conn = sqlite3.connect('instance/chestai.db'); conn.execute('INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)', ('dr_new', generate_password_hash('pass123'), 'doctor')); conn.commit(); conn.close(); print('User added.')"


#to stop proc
#Stop-Process -Name python -Force
