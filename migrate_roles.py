import sqlite3, os

DB = os.path.join(os.path.dirname(__file__), "instance", "chestai.db")
conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("""CREATE TABLE users_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('doctor', 'admin')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)""")
cur.execute("INSERT INTO users_new SELECT * FROM users")
cur.execute("DROP TABLE users")
cur.execute("ALTER TABLE users_new RENAME TO users")
conn.commit()
conn.close()
print("Migration complete — role constraint updated to (doctor, admin).")
