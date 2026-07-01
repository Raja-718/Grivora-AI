"""
auth_system/auth_routes.py
All login / register / logout / OTP / Google OAuth routes.
Registered as Flask Blueprint — mounted in run.py.

GOOGLE OAUTH SETUP (one-time):
1. Go to console.cloud.google.com
2. APIs & Services → Credentials → Create OAuth 2.0 Client ID
3. Application type: Web application
4. Authorized redirect URIs:
     http://localhost:5000/auth/google/callback
5. Copy Client ID and Secret → add to .env:
     GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
     GOOGLE_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxxx
"""
from flask import (Blueprint, render_template, request,
                   redirect, url_for, session, jsonify)
from auth_system.user_store import (find_by_email, find_by_mobile,
                                    create_user, update_last_login)
from auth_system.otp_store  import generate as gen_otp, verify as verify_otp
from auth_system.otp_sender import send_email_otp, send_mobile_otp
from app import limiter
import re, os, json, secrets, urllib.request, urllib.parse

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# ── helpers ───────────────────────────────────────────────────────────────────

def is_valid_email(e: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e))

def is_valid_mobile(m: str) -> bool:
    return bool(re.match(r"^\d{10}$", m))

def logged_in() -> bool:
    return "user_id" in session

def _google_configured() -> bool:
    from dotenv import load_dotenv; load_dotenv(override=True)
    return bool(os.getenv("GOOGLE_CLIENT_ID","").strip())

# ── register ──────────────────────────────────────────────────────────────────

@auth_bp.route("/register", methods=["GET"])
def register():
    if logged_in():
        return redirect("/")
    google_ok = _google_configured()
    return render_template("auth/register.html", google_ok=google_ok)

@auth_bp.route("/register/send-otp", methods=["POST"])
@limiter.limit("5 per minute; 20 per hour")
def register_send_otp():
    data   = request.get_json() or {}
    name   = data.get("name","").strip()
    email  = data.get("email","").strip().lower()
    mobile = re.sub(r"\D","", data.get("mobile",""))

    if not name:
        return jsonify({"ok": False, "msg": "Full name is required."})
    if not is_valid_email(email):
        return jsonify({"ok": False, "msg": "Enter a valid email address."})
    if not is_valid_mobile(mobile):
        return jsonify({"ok": False, "msg": "Enter a valid 10-digit mobile number."})
    if find_by_email(email):
        return jsonify({"ok": False, "msg": "Email already registered. Please sign in."})
    if find_by_mobile(mobile):
        return jsonify({"ok": False, "msg": "Mobile already registered. Please sign in."})

    session["pending_reg"] = {"name": name, "email": email, "mobile": mobile}
    otp  = gen_otp(email)
    ok, info = send_email_otp(email, otp, name)
    if not ok:
        return jsonify({"ok": False, "msg": f"Could not send OTP: {info}"})
    return jsonify({"ok": True, "msg": f"OTP sent to {email}"})

@auth_bp.route("/register/verify-otp", methods=["POST"])
@limiter.limit("10 per minute")
def register_verify_otp():
    data = request.get_json() or {}
    code = data.get("otp","").strip()
    reg  = session.get("pending_reg")
    if not reg:
        return jsonify({"ok": False, "msg": "Session expired. Please start again."})

    ok, reason = verify_otp(reg["email"], code)
    if not ok:
        return jsonify({"ok": False, "msg": {
            "not_found":        "No OTP found. Please resend.",
            "expired":          "OTP expired. Please resend.",
            "invalid":          "Incorrect code. Please try again.",
            "too_many_attempts":"Too many attempts. Please resend.",
        }.get(reason, "Invalid OTP.")})

    user = create_user(reg["name"], reg["email"], reg["mobile"])
    update_last_login(user["id"])
    session.pop("pending_reg", None)
    session["user_id"]    = user["id"]
    session["user_name"]  = user["name"]
    session["user_email"] = user["email"]
    return jsonify({"ok": True, "msg": "Account created!", "redirect": "/"})

# ── login ─────────────────────────────────────────────────────────────────────

@auth_bp.route("/login", methods=["GET"])
def login():
    if logged_in():
        return redirect("/")
    google_ok    = _google_configured()
    google_error = request.args.get("google_error","")
    error_msgs = {
        "not_configured": "Google Sign-In is not configured yet.",
        "cancelled":      "Google Sign-In was cancelled.",
        "token_failed":   "Google authentication failed. Please try again.",
        "no_email":       "Could not retrieve your Google email.",
        "server_error":   "A server error occurred during Google Sign-In.",
    }
    google_msg = error_msgs.get(google_error,"")
    return render_template("auth/login.html",
                           google_ok=google_ok, google_msg=google_msg)

@auth_bp.route("/login/send-otp", methods=["POST"])
@limiter.limit("5 per minute; 20 per hour")
def login_send_otp():
    data       = request.get_json() or {}
    identifier = data.get("identifier","").strip().lower()
    via        = data.get("via","email")

    if via == "email":
        if not is_valid_email(identifier):
            return jsonify({"ok": False, "msg": "Enter a valid email address."})
        user = find_by_email(identifier)
        if not user:
            return jsonify({"ok": False, "msg": "No account found. Please register first."})
        otp = gen_otp(identifier)
        ok, info = send_email_otp(identifier, otp, user["name"])
        label = identifier
    else:
        mobile = re.sub(r"\D","", identifier)
        if not is_valid_mobile(mobile):
            return jsonify({"ok": False, "msg": "Enter a valid 10-digit mobile number."})
        user = find_by_mobile(mobile)
        if not user:
            return jsonify({"ok": False, "msg": "No account found with this number. Please register."})
        otp = gen_otp(mobile)
        ok, info = send_mobile_otp(mobile, otp)
        identifier = mobile
        label = f"mobile ****{mobile[-4:]}"

    if not ok:
        return jsonify({"ok": False, "msg": f"Could not send OTP: {info}"})

    session["login_key"] = identifier
    session["login_via"] = via
    return jsonify({"ok": True, "msg": f"OTP sent to {label}"})

@auth_bp.route("/login/verify-otp", methods=["POST"])
@limiter.limit("10 per minute")
def login_verify_otp():
    data = request.get_json() or {}
    code = data.get("otp","").strip()
    key  = session.get("login_key")
    via  = session.get("login_via","email")

    if not key:
        return jsonify({"ok": False, "msg": "Session expired. Please start again."})

    ok, reason = verify_otp(key, code)
    if not ok:
        return jsonify({"ok": False, "msg": {
            "not_found":        "No OTP found. Please resend.",
            "expired":          "OTP expired. Please resend.",
            "invalid":          "Incorrect code. Please try again.",
            "too_many_attempts":"Too many attempts. Please resend.",
        }.get(reason, "Invalid OTP.")})

    user = find_by_email(key) if via == "email" else find_by_mobile(key)
    if not user:
        return jsonify({"ok": False, "msg": "User not found."})

    update_last_login(user["id"])
    session.pop("login_key", None)
    session.pop("login_via", None)
    session["user_id"]    = user["id"]
    session["user_name"]  = user["name"]
    session["user_email"] = user["email"]
    return jsonify({"ok": True, "msg": "Welcome back!", "redirect": "/"})

# ── logout ────────────────────────────────────────────────────────────────────

@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ── resend OTP ────────────────────────────────────────────────────────────────

@auth_bp.route("/resend-otp", methods=["POST"])
@limiter.limit("3 per minute; 10 per hour")
def resend_otp():
    data = request.get_json() or {}
    ctx  = data.get("ctx","login")

    if ctx == "register":
        reg = session.get("pending_reg")
        if not reg:
            return jsonify({"ok": False, "msg": "Session expired. Please start over."})
        otp = gen_otp(reg["email"])
        ok, _ = send_email_otp(reg["email"], otp, reg["name"])
        return jsonify({"ok": ok, "msg": "OTP resent to your email." if ok else "Failed to resend."})
    else:
        key = session.get("login_key")
        via = session.get("login_via","email")
        if not key:
            return jsonify({"ok": False, "msg": "Session expired. Please start over."})
        otp = gen_otp(key)
        if via == "email":
            user = find_by_email(key)
            ok, _ = send_email_otp(key, otp, user["name"] if user else "")
        else:
            ok, _ = send_mobile_otp(key, otp)
        return jsonify({"ok": ok, "msg": "OTP resent." if ok else "Failed to resend."})

# ── current user (JSON) ───────────────────────────────────────────────────────

@auth_bp.route("/me")
def me():
    if not logged_in():
        return jsonify({"logged_in": False})
    return jsonify({
        "logged_in": True,
        "name":  session.get("user_name",""),
        "email": session.get("user_email",""),
    })

# ── Google OAuth ──────────────────────────────────────────────────────────────

@auth_bp.route("/google")
def google_login():
    from dotenv import load_dotenv; load_dotenv(override=True)
    client_id = os.getenv("GOOGLE_CLIENT_ID","").strip()
    if not client_id:
        return redirect("/auth/login?google_error=not_configured")

    # CSRF protection: generate a random state token and store in session.
    # The callback must receive the same value or we reject the login.
    state = secrets.token_urlsafe(32)
    session["google_oauth_state"] = state

    redirect_uri = request.host_url.rstrip("/") + "/auth/google/callback"
    params = {
        "client_id":     client_id,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "select_account",
        "state":         state,
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    return redirect(url)

@auth_bp.route("/google/callback")
def google_callback():
    from dotenv import load_dotenv; load_dotenv(override=True)
    code            = request.args.get("code","")
    error           = request.args.get("error","")
    returned_state  = request.args.get("state","")
    expected_state  = session.pop("google_oauth_state", None)

    if error or not code:
        return redirect("/auth/login?google_error=cancelled")

    # Verify state to block OAuth CSRF. Both sides must match exactly.
    if not expected_state or not returned_state or \
       not secrets.compare_digest(expected_state, returned_state):
        print(f"[GOOGLE AUTH] State mismatch — rejecting callback", flush=True)
        return redirect("/auth/login?google_error=token_failed")

    client_id     = os.getenv("GOOGLE_CLIENT_ID","").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET","").strip()
    redirect_uri  = request.host_url.rstrip("/") + "/auth/google/callback"

    try:
        # Exchange code for access token
        token_body = urllib.parse.urlencode({
            "code":          code,
            "client_id":     client_id,
            "client_secret": client_secret,
            "redirect_uri":  redirect_uri,
            "grant_type":    "authorization_code",
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=token_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            token_data = json.loads(resp.read().decode())

        access_token = token_data.get("access_token","")
        if not access_token:
            return redirect("/auth/login?google_error=token_failed")

        # Fetch user profile
        profile_req = urllib.request.Request(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        with urllib.request.urlopen(profile_req, timeout=15) as resp:
            user_info = json.loads(resp.read().decode())

        email = user_info.get("email","").lower().strip()
        name  = user_info.get("name","").strip() or email.split("@")[0].title()

        if not email:
            return redirect("/auth/login?google_error=no_email")

        # Find or create user
        user = find_by_email(email)
        if not user:
            user = create_user(name, email, "")   # Google users have no mobile

        update_last_login(user["id"])
        session["user_id"]    = user["id"]
        session["user_name"]  = user["name"]
        session["user_email"] = user["email"]
        session["auth_via"]   = "google"

        print(f"[GOOGLE AUTH] Logged in: {email}", flush=True)
        return redirect("/")

    except Exception as e:
        print(f"[GOOGLE AUTH ERROR] {type(e).__name__}: {e}", flush=True)
        return redirect("/auth/login?google_error=server_error")
