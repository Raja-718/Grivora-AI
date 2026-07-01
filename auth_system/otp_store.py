"""
auth_system/otp_store.py
SQLite-backed OTP store with 5-minute expiry.
Compatible with Python 3.9+

Why SQLite instead of in-memory:
- Survives app restarts (user's active OTP isn't wiped every deploy)
- Works correctly across multi-worker deployments (gunicorn -w 4)
- Zero extra dependencies (sqlite3 ships with CPython)
- File lock handles concurrent access safely for our low write volume

Public API is unchanged:
    generate(key)   -> 6-digit code
    verify(key, code) -> (ok, reason)
    has_pending(key)
    get_otp_for_testing(key)
"""
import os
import random
import sqlite3
import time
import threading
from typing import Tuple

OTP_TTL      = 300   # seconds (5 min)
MAX_ATTEMPTS = 5

DB_PATH = os.path.join(os.path.dirname(__file__), "otp_store.db")

# SQLite connections are not thread-safe by default; serialize access.
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    """Open a short-lived connection. Each call gets its own connection
    so we don't hold FDs open across request boundaries."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")      # concurrent readers
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _init_db():
    """Create the OTP table on first use."""
    with _lock, _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS otps (
                key         TEXT PRIMARY KEY,
                code        TEXT NOT NULL,
                expires_at  REAL NOT NULL,
                attempts    INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()


_init_db()


def _purge_expired(conn: sqlite3.Connection):
    """Remove expired rows. Cheap; runs before each operation."""
    conn.execute("DELETE FROM otps WHERE expires_at < ?", (time.time(),))


def generate(key: str) -> str:
    """Generate and store a 6-digit OTP for key (email or mobile)."""
    code = str(random.randint(100000, 999999))
    expires_at = time.time() + OTP_TTL

    with _lock, _connect() as conn:
        _purge_expired(conn)
        # Upsert: replace any existing OTP for this key
        conn.execute("""
            INSERT INTO otps (key, code, expires_at, attempts)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(key) DO UPDATE SET
                code       = excluded.code,
                expires_at = excluded.expires_at,
                attempts   = 0
        """, (key, code, expires_at))
        conn.commit()

    # Dev console output — gated behind DEBUG so production logs stay clean.
    if os.getenv("DEBUG", "").lower() in ("true", "1", "yes"):
        print(f"\n{'='*55}")
        print(f"  [OTP GENERATED] key={key}  code={code}")
        print(f"{'='*55}\n")

    return code


def verify(key: str, code: str) -> Tuple[bool, str]:
    """
    Returns (True, 'ok') or (False, reason).
    reason: 'not_found' | 'expired' | 'invalid' | 'too_many_attempts'
    """
    with _lock, _connect() as conn:
        _purge_expired(conn)
        row = conn.execute(
            "SELECT code, expires_at, attempts FROM otps WHERE key = ?",
            (key,)
        ).fetchone()

        if row is None:
            return False, "not_found"

        stored_code, expires_at, attempts = row

        if time.time() > expires_at:
            conn.execute("DELETE FROM otps WHERE key = ?", (key,))
            conn.commit()
            return False, "expired"

        if attempts >= MAX_ATTEMPTS:
            conn.execute("DELETE FROM otps WHERE key = ?", (key,))
            conn.commit()
            return False, "too_many_attempts"

        if code != stored_code:
            conn.execute(
                "UPDATE otps SET attempts = attempts + 1 WHERE key = ?",
                (key,)
            )
            conn.commit()
            return False, "invalid"

        # Success — OTP is single-use, delete it.
        conn.execute("DELETE FROM otps WHERE key = ?", (key,))
        conn.commit()
        return True, "ok"


def has_pending(key: str) -> bool:
    """True if a non-expired OTP exists for this key."""
    with _lock, _connect() as conn:
        _purge_expired(conn)
        row = conn.execute(
            "SELECT 1 FROM otps WHERE key = ? AND expires_at > ?",
            (key, time.time())
        ).fetchone()
        return row is not None


def get_otp_for_testing(key: str) -> str:
    """DEV helper — returns current OTP if still valid. Empty string otherwise."""
    with _lock, _connect() as conn:
        _purge_expired(conn)
        row = conn.execute(
            "SELECT code FROM otps WHERE key = ? AND expires_at > ?",
            (key, time.time())
        ).fetchone()
        return row[0] if row else ""
