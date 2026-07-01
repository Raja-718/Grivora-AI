"""
run.py — Grivora AI entry point
Registers main blueprint + auth blueprint + auth middleware.
"""
from app import create_app
from auth_system.auth_routes     import auth_bp
from auth_system.auth_middleware import register_auth_guard

app = create_app()

# ── Register blueprints ───────────────────────────────────────────────────────
app.register_blueprint(auth_bp)

# ── Attach login guard (protects /upload, /chat, /dashboard, etc.) ───────────
register_auth_guard(app)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
