from functools import wraps

from flask import flash, g, redirect, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from db import get_db


def create_user(username, password, role):
    if role not in ("doctor", "admin"):
        raise ValueError(f"invalid role: {role}")
    conn = get_db()
    conn.execute(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        (username, generate_password_hash(password), role),
    )
    conn.commit()
    conn.close()


def verify_user(username, password):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if user is None or not check_password_hash(user["password_hash"], password):
        return None
    return user


def load_logged_in_user():
    user_id = session.get("user_id")
    if user_id is None:
        g.user = None
    else:
        conn = get_db()
        g.user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            flash("Please log in to continue.")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def role_required(*roles):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if g.user is None:
                flash("Please log in to continue.")
                return redirect(url_for("login"))
            if g.user["role"] not in roles:
                flash("You do not have permission to view this page.")
                dest = "admin_panel" if g.user["role"] == "admin" else "dashboard"
                return redirect(url_for(dest))
            return view(*args, **kwargs)
        return wrapped
    return decorator
