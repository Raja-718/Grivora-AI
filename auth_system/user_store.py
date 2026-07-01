"""
auth_system/user_store.py
SQLite-backed user store.

Public API (unchanged):
    find_by_email(email)
    find_by_mobile(mobile)
    find_by_id(user_id)
    create_user(name, email, mobile)
    update_last_login(user_id)
    get_all_users()

Why SQLite instead of a JSON flat file:
- Safe under concurrent writes (two simultaneous registrations can't clobber
  each other, which the flat file allowed).
- Works across multiple gunicorn workers.
- Zero extra dependencies (sqlite3 ships with CPython).
- Indexed lookups on email/mobile remain fast as users grow.

One-time migration: if auth_system/users.json exists, its contents are
imported on first use. The JSON file is then renamed to users.json.migrated
so it's obvious no further writes go there.
"""
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Optional, Dict, List

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")
JSON_LEGACY_PATH = os.path.join(os.path.dirname(__file__), "users.json")

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict]:
    if row is None:
        return None
    return {
        "id":         row["id"],
        "name":       row["name"],
        "email":      row["email"],
        "mobile":     row["mobile"] or "",
        "created_at": row["created_at"],
        "last_login": row["last_login"],
    }


def _init_db():
    """Create the users table and migrate from users.json if present."""
    with _lock, _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                email       TEXT NOT NULL UNIQUE COLLATE NOCASE,
                mobile      TEXT,
                created_at  TEXT NOT NULL,
                last_login  TEXT
            )
        """)
        # Mobile is optional (Google OAuth users) but when set must be unique.
        # Partial unique index skips NULL/empty rows.
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_mobile
            ON users(mobile) WHERE mobile IS NOT NULL AND mobile != ''
        """)
        conn.commit()

        # One-time migration from users.json.
        if os.path.exists(JSON_LEGACY_PATH):
            try:
                with open(JSON_LEGACY_PATH, "r", encoding="utf-8") as f:
                    legacy = json.load(f)
                migrated = 0
                for u in legacy.get("users", []):
                    try:
                        conn.execute("""
                            INSERT OR IGNORE INTO users
                            (id, name, email, mobile, created_at, last_login)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            u.get("id") or str(uuid.uuid4()),
                            u.get("name", ""),
                            u.get("email", "").lower(),
                            u.get("mobile", ""),
                            u.get("created_at") or datetime.utcnow().isoformat(),
                            u.get("last_login"),
                        ))
                        migrated += 1
                    except Exception:
                        continue
                conn.commit()

                # Rename the old file so it's not confusing going forward.
                backup = JSON_LEGACY_PATH + ".migrated"
                if not os.path.exists(backup):
                    os.rename(JSON_LEGACY_PATH, backup)
                if migrated:
                    print(f"[USER_STORE] Migrated {migrated} users from users.json",
                          flush=True)
            except Exception as e:
                print(f"[USER_STORE] Legacy JSON migration skipped: {e}", flush=True)


_init_db()


def find_by_email(email: str) -> Optional[Dict]:
    if not email:
        return None
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
            (email.strip().lower(),)
        ).fetchone()
        return _row_to_dict(row)


def find_by_mobile(mobile: str) -> Optional[Dict]:
    if not mobile:
        return None
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE mobile = ?",
            (mobile.strip(),)
        ).fetchone()
        return _row_to_dict(row)


def find_by_id(user_id: str) -> Optional[Dict]:
    if not user_id:
        return None
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return _row_to_dict(row)


def create_user(name: str, email: str, mobile: str) -> Dict:
    user = {
        "id":         str(uuid.uuid4()),
        "name":       name.strip(),
        "email":      email.strip().lower(),
        "mobile":     mobile.strip() if mobile else "",
        "created_at": datetime.utcnow().isoformat(),
        "last_login": None,
    }
    with _lock, _connect() as conn:
        conn.execute("""
            INSERT INTO users (id, name, email, mobile, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user["id"], user["name"], user["email"],
            user["mobile"] or None,     # store NULL for empty mobile so the
                                         # partial unique index is happy
            user["created_at"], user["last_login"],
        ))
        conn.commit()
    # Normalize: API has always returned mobile as string, never None.
    user["mobile"] = user["mobile"] or ""
    return user


def update_last_login(user_id: str):
    if not user_id:
        return
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), user_id)
        )
        conn.commit()


def get_all_users() -> List[Dict]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
