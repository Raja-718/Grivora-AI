"""
app/__init__.py
Grivora AI application factory.

Creates the Flask app, attaches config, rate limiter (for LLM endpoints),
CSRF protection, and registers blueprints.
"""
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect

from config import Config

# ── Module-level extensions (attached to app in create_app) ────────────────
# Rate limiter: default in-memory backend.
# For multi-worker production deployments, set limiter.storage_uri to a
# shared backend (Redis: "redis://localhost:6379") in config.py.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],      # No global default — per-route decorators only.
    headers_enabled=True,   # Send X-RateLimit-* headers so clients can back off.
)

# ── CSRF Protection ────────────────────────────────────────────────────────
# Security rationale:
#   - All /api/* endpoints communicate via JSON (Content-Type: application/json).
#     Browsers CANNOT submit JSON payloads via cross-origin <form> or <img> tags,
#     which is the attack vector CSRF tokens protect against.
#   - Session cookies use SameSite=Lax (set in config.py), which blocks
#     cross-site POST requests entirely on modern browsers.
#   - All /api/* endpoints additionally require an authenticated session
#     (enforced by auth_middleware.py before_request guard).
#   - Auth form routes (/auth/register, /auth/login) retain full CSRF
#     protection since they accept traditional form-encoded POST data.
#
# Therefore: /api/* routes are explicitly exempt from CSRF tokens.
# Auth routes and any future HTML-form POST routes retain full protection.
csrf = CSRFProtect()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Attach extensions
    limiter.init_app(app)
    csrf.init_app(app)

    from app.routes import main
    app.register_blueprint(main)

    # ── Exempt JSON API routes from CSRF ───────────────────────────────
    # See security rationale above. This covers all /api/* endpoints in the
    # main blueprint (PMA, BIA, chart-data, upload, etc.).
    csrf.exempt(main)

    return app
