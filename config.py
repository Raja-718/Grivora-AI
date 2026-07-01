# config.py - App configuration and environment variables
import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


def _env_bool(key: str, default: bool) -> bool:
    """Parse a boolean env var safely. Accepts true/1/yes (case-insensitive)."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


class Config:
    # ── Core ────────────────────────────────────────────────────────
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload

    # DEBUG defaults to FALSE for safety. Set DEBUG=True explicitly in .env
    # for local development. Never ship with DEBUG=True — it exposes the
    # Werkzeug debugger which allows arbitrary code execution.
    DEBUG = _env_bool("DEBUG", False)

    # File types accepted by /api/upload (enforced in routes.py).
    # Matches what src/data_loader.py can actually parse.
    ALLOWED_EXTENSIONS = {
        "csv", "tsv", "tab", "txt", "dat",
        "xlsx", "xls", "xlsb", "xlsm", "ods",
        "json", "parquet", "pq", "xml",
    }

    # ── Session / cookie hardening ──────────────────────────────────
    # Cookie-based sessions (default). We keep blobs out of the session
    # so the 4KB cookie limit is never a concern.
    SESSION_COOKIE_SECURE   = _env_bool("SESSION_COOKIE_SECURE", not DEBUG)
    SESSION_COOKIE_HTTPONLY = True        # JS can't read the cookie
    SESSION_COOKIE_SAMESITE = "Lax"       # CSRF baseline; "Strict" breaks OAuth redirects
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)

    # ── CSRF (Flask-WTF) ────────────────────────────────────────────
    # We keep CSRFProtect available but disable the default auto-check so
    # our JSON API (which doesn't send form tokens) still works. Auth form
    # POSTs and any future @csrf.protect'ed views are still covered, plus
    # SameSite=Lax gives us baseline CSRF protection for everything else.
    WTF_CSRF_ENABLED       = True
    WTF_CSRF_CHECK_DEFAULT = False
    WTF_CSRF_TIME_LIMIT    = 3600          # 1 hour
