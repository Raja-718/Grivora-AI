"""
auth_system/auth_middleware.py
Login-required decorator + Flask before_request guard.
Attach to the main Flask app in run.py.

Protected routes: /upload, /chat, /dashboard, /data-preview,
                  /analysis-dashboard, /analysis, /predict, /bi,
                  /api/upload, /api/chat, /api/chart-data, /api/preview-data
"""
from flask import session, redirect, request, jsonify
from functools import wraps

# Routes that require login
PROTECTED_PREFIXES = (
    "/upload",
    "/chat",
    "/dashboard",
    "/data-preview",
    "/auto-dashboard",
    "/analysis-dashboard",
    "/analysis",
    "/predict",
    "/bi",
    "/api/upload",
    "/api/chat",
    "/api/chart-data",
    "/api/preview-data",
    "/api/edit-data",
    "/api/session-file",
    "/api/auto-dashboard",
    "/api/pma/",
    "/api/bia/",
)

# Auth routes + static assets are always public
PUBLIC_PREFIXES = (
    "/auth/",
    "/static/",
    "/favicon",
)


def is_public(path: str) -> bool:
    if path == "/":
        return True
    for p in PUBLIC_PREFIXES:
        if path.startswith(p):
            return True
    return False


def is_protected(path: str) -> bool:
    for p in PROTECTED_PREFIXES:
        if path.startswith(p):
            return True
    return False


def register_auth_guard(app):
    """
    Call this in run.py after creating the app:
        from auth_system.auth_middleware import register_auth_guard
        register_auth_guard(app)
    """
    @app.before_request
    def auth_guard():
        path = request.path
        if is_public(path):
            return None
        if not is_protected(path):
            return None

        # Logged in — allow
        if "user_id" in session:
            return None

        # API routes — return JSON 401
        if path.startswith("/api/"):
            return jsonify({
                "error": "Authentication required.",
                "login_url": "/auth/login"
            }), 401

        # Page routes — redirect to login with next param
        return redirect(f"/auth/login?next={path}")


def login_required(f):
    """Decorator for individual route functions (optional use)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required.", "login_url": "/auth/login"}), 401
            return redirect(f"/auth/login?next={request.path}")
        return f(*args, **kwargs)
    return decorated
