"""
auth_system/otp_sender.py
─────────────────────────────────────────────────────────────────────────────
PRODUCTION EMAIL & MOBILE OTP SENDER

EMAIL PROVIDERS (tried in order):
  1. Gmail SMTP  — set SMTP_USER + SMTP_PASSWORD (Gmail App Password)
  2. SendGrid    — set SENDGRID_API_KEY
  3. Mailgun     — set MAILGUN_API_KEY + MAILGUN_DOMAIN
  4. DEV fallback — prints OTP to terminal (never reaches production users)

MOBILE PROVIDERS:
  1. Twilio SMS  — set TWILIO_SID + TWILIO_TOKEN + TWILIO_FROM
  2. DEV fallback — prints OTP to terminal

─────────────────────────────────────────────────────────────────────────────
QUICKEST SETUP (Gmail App Password):
  1. Enable 2-Step Verification at: myaccount.google.com/security
  2. Go to: myaccount.google.com/apppasswords
  3. App name: GrivoraAI → click Generate
  4. Copy the 16-char password shown
  5. Add to .env:
       SMTP_USER=your_gmail@gmail.com
       SMTP_PASSWORD=abcdefghijklmnop     ← no spaces
─────────────────────────────────────────────────────────────────────────────
"""

import smtplib, os, sys, ssl, urllib.request, urllib.parse, json as _json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Tuple

APP_NAME     = "Grivora AI"
APP_COLOR    = "#1d4ed8"
SUPPORT_MAIL = os.getenv("SUPPORT_EMAIL", "support@grivora.ai")


# ─────────────────────────────────────────────────────────────────────────────
#  ENV LOADER — always reads fresh so no restart needed after editing .env
# ─────────────────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    """Read a .env key fresh each call."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    return os.getenv(key, default).strip()


# ─────────────────────────────────────────────────────────────────────────────
#  HTML EMAIL TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

def _build_html(otp: str, name: str, purpose: str = "verification") -> str:
    greeting = f"Hi {name}," if name else "Hello,"
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{APP_NAME} OTP</title></head>
<body style="margin:0;padding:0;background:#f0f4ff;font-family:'Segoe UI',Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f0f4ff;padding:40px 16px;">
  <tr><td align="center">
    <table width="520" cellpadding="0" cellspacing="0"
           style="background:#ffffff;border-radius:20px;overflow:hidden;
                  box-shadow:0 4px 32px rgba(29,78,216,0.12);">
      <!-- Header -->
      <tr><td style="background:linear-gradient(135deg,#1d4ed8 0%,#7c3aed 55%,#0891b2 100%);
                     padding:28px 36px;text-align:center;">
        <div style="width:52px;height:52px;background:rgba(255,255,255,0.18);
                    border-radius:14px;margin:0 auto 10px;line-height:52px;font-size:26px;">👁</div>
        <span style="font-size:22px;font-weight:800;color:#ffffff;letter-spacing:-0.02em;">
          {APP_NAME}
        </span>
      </td></tr>
      <!-- Body -->
      <tr><td style="padding:36px 36px 28px;">
        <p style="margin:0 0 6px;font-size:15px;color:#0a1628;">{greeting}</p>
        <p style="margin:0 0 24px;font-size:14px;color:#3a5080;line-height:1.65;">
          Your {purpose} code for <strong>{APP_NAME}</strong> is ready.
          Enter it within <strong>5 minutes</strong>.
        </p>
        <!-- OTP Box -->
        <div style="text-align:center;margin:0 0 28px;">
          <div style="display:inline-block;font-size:46px;font-weight:800;
                      letter-spacing:14px;color:#1d4ed8;
                      background:#eef4ff;padding:20px 36px;
                      border-radius:16px;border:2px solid #dde8ff;">
            {otp}
          </div>
        </div>
        <!-- Info box -->
        <div style="background:#fafbff;border:1px solid #e4eaf8;border-radius:12px;
                    padding:14px 18px;margin-bottom:24px;">
          <table width="100%" cellpadding="0" cellspacing="0">
            <tr>
              <td width="20" style="vertical-align:top;padding-top:2px;font-size:16px;">⏱</td>
              <td style="font-size:12.5px;color:#7a96c0;padding-left:8px;line-height:1.6;">
                <strong style="color:#3a5080;">Expires in 5 minutes.</strong>
                Never share this code with anyone — {APP_NAME} staff will never ask for it.
              </td>
            </tr>
          </table>
        </div>
        <p style="margin:0;font-size:12px;color:#b0c2dc;text-align:center;line-height:1.6;">
          If you didn't request this code, please ignore this email or contact us at
          <a href="mailto:{SUPPORT_MAIL}" style="color:#7a96c0;">{SUPPORT_MAIL}</a>
        </p>
      </td></tr>
      <!-- Footer -->
      <tr><td style="background:#f8faff;border-top:1px solid #e4eaf8;
                     padding:16px 36px;text-align:center;">
        <p style="margin:0;font-size:11px;color:#b0c2dc;">
          © 2025 {APP_NAME} · AI Data Intelligence Platform
        </p>
      </td></tr>
    </table>
  </td></tr>
</table>
</body></html>"""


def _build_plain(otp: str, name: str) -> str:
    greeting = f"Hi {name}," if name else "Hello,"
    return (
        f"{greeting}\n\n"
        f"Your {APP_NAME} verification code is: {otp}\n\n"
        f"This code expires in 5 minutes. Do not share it with anyone.\n\n"
        f"© 2025 {APP_NAME}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TERMINAL PRINTER — always runs so dev can see OTP during testing
# ─────────────────────────────────────────────────────────────────────────────

def _print_otp(channel: str, identifier: str, otp: str):
    b = "═" * 58
    print(f"\n{b}", flush=True)
    print(f"  [{APP_NAME.upper()}]  {channel.upper()} OTP", flush=True)
    print(f"  To      :  {identifier}", flush=True)
    print(f"  OTP     :  {otp}", flush=True)
    print(f"  Expires :  5 minutes", flush=True)
    print(f"{b}\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PROVIDER 1 — Gmail SMTP (most common for small projects)
# ─────────────────────────────────────────────────────────────────────────────

def _send_gmail(to_email: str, subject: str, html: str, plain: str) -> Tuple[bool, str]:
    host  = _env("SMTP_HOST", "smtp.gmail.com")
    port  = int(_env("SMTP_PORT", "587"))
    user  = _env("SMTP_USER")
    pwd   = _env("SMTP_PASSWORD")

    if not user or not pwd:
        return False, "not_configured"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{APP_NAME} <{user}>"
        msg["To"]      = to_email
        msg["Reply-To"] = SUPPORT_MAIL
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))

        ctx = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=20) as srv:
            srv.ehlo()
            srv.starttls(context=ctx)
            srv.ehlo()
            srv.login(user, pwd)
            srv.sendmail(user, to_email, msg.as_string())

        print(f"[GMAIL ✓] Sent to {to_email}", flush=True)
        return True, "sent_gmail"

    except smtplib.SMTPAuthenticationError:
        msg = (
            "Gmail authentication failed. "
            "You need an App Password (not your regular password). "
            "Setup: myaccount.google.com → Security → App passwords"
        )
        print(f"[GMAIL ✗] {msg}", flush=True)
        return False, msg

    except smtplib.SMTPConnectError as e:
        print(f"[GMAIL ✗] Connect error: {e}", flush=True)
        return False, f"SMTP connect failed: {e}"

    except smtplib.SMTPRecipientsRefused:
        print(f"[GMAIL ✗] Recipient refused: {to_email}", flush=True)
        return False, f"Email address rejected: {to_email}"

    except Exception as e:
        print(f"[GMAIL ✗] {type(e).__name__}: {e}", flush=True)
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  PROVIDER 2 — SendGrid REST API (free 100 emails/day)
# ─────────────────────────────────────────────────────────────────────────────

def _send_sendgrid(to_email: str, subject: str, html: str, plain: str) -> Tuple[bool, str]:
    api_key  = _env("SENDGRID_API_KEY")
    from_email = _env("SENDGRID_FROM", _env("SMTP_USER", f"noreply@grivora.ai"))

    if not api_key:
        return False, "not_configured"

    payload = _json.dumps({
        "personalizations": [{"to": [{"email": to_email}]}],
        "from":    {"email": from_email, "name": APP_NAME},
        "subject": subject,
        "content": [
            {"type": "text/plain", "value": plain},
            {"type": "text/html",  "value": html},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"[SENDGRID ✓] Sent to {to_email} — status {resp.status}", flush=True)
            return True, "sent_sendgrid"
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[SENDGRID ✗] HTTP {e.code}: {body[:200]}", flush=True)
        return False, f"SendGrid HTTP {e.code}"
    except Exception as e:
        print(f"[SENDGRID ✗] {type(e).__name__}: {e}", flush=True)
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  PROVIDER 3 — Mailgun REST API (free 1 000 emails/month)
# ─────────────────────────────────────────────────────────────────────────────

def _send_mailgun(to_email: str, subject: str, html: str, plain: str) -> Tuple[bool, str]:
    api_key = _env("MAILGUN_API_KEY")
    domain  = _env("MAILGUN_DOMAIN")          # e.g. mg.yourdomain.com
    region  = _env("MAILGUN_REGION", "us")    # "us" or "eu"
    from_addr = _env("MAILGUN_FROM", f"noreply@{domain}" if domain else "")

    if not api_key or not domain:
        return False, "not_configured"

    base = "https://api.eu.mailgun.net" if region == "eu" else "https://api.mailgun.net"
    url  = f"{base}/v3/{domain}/messages"

    body = urllib.parse.urlencode({
        "from":    f"{APP_NAME} <{from_addr}>",
        "to":      to_email,
        "subject": subject,
        "text":    plain,
        "html":    html,
    }).encode("utf-8")

    import base64
    creds = base64.b64encode(f"api:{api_key}".encode()).decode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Authorization": f"Basic {creds}"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"[MAILGUN ✓] Sent to {to_email}", flush=True)
            return True, "sent_mailgun"
    except urllib.error.HTTPError as e:
        print(f"[MAILGUN ✗] HTTP {e.code}", flush=True)
        return False, f"Mailgun HTTP {e.code}"
    except Exception as e:
        print(f"[MAILGUN ✗] {type(e).__name__}: {e}", flush=True)
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def send_email_otp(to_email: str, otp: str, name: str = "") -> Tuple[bool, str]:
    """
    Send OTP email using first available provider.
    Priority: Gmail SMTP → SendGrid → Mailgun → terminal fallback.
    Always prints OTP to terminal as a safety net.
    """
    # Always print to terminal (dev safety net)
    _print_otp("EMAIL", to_email, otp)

    subject = f"{otp} — Your {APP_NAME} verification code"
    html    = _build_html(otp, name)
    plain   = _build_plain(otp, name)

    # ── Try Gmail SMTP first ──────────────────────────────────────────────
    ok, info = _send_gmail(to_email, subject, html, plain)
    if ok:
        return True, info
    if info != "not_configured":
        # Gmail is configured but failed — try fallbacks
        print(f"[EMAIL] Gmail failed ({info}), trying next provider…", flush=True)

    # ── Try SendGrid ──────────────────────────────────────────────────────
    ok, info = _send_sendgrid(to_email, subject, html, plain)
    if ok:
        return True, info
    if info != "not_configured":
        print(f"[EMAIL] SendGrid failed ({info}), trying next provider…", flush=True)

    # ── Try Mailgun ───────────────────────────────────────────────────────
    ok, info = _send_mailgun(to_email, subject, html, plain)
    if ok:
        return True, info
    if info != "not_configured":
        print(f"[EMAIL] Mailgun failed ({info})", flush=True)

    # ── No provider configured — dev mode ─────────────────────────────────
    configured = any([
        _env("SMTP_USER") and _env("SMTP_PASSWORD"),
        _env("SENDGRID_API_KEY"),
        _env("MAILGUN_API_KEY") and _env("MAILGUN_DOMAIN"),
    ])
    if not configured:
        print("[EMAIL] ⚠  No email provider configured. OTP shown in terminal only.", flush=True)
        # Return True so OTP flow continues — user gets code from terminal in dev
        return True, "dev_console"

    # All providers configured but all failed
    return False, "All configured email providers failed. Check terminal logs."


# ─────────────────────────────────────────────────────────────────────────────
#  MOBILE OTP
# ─────────────────────────────────────────────────────────────────────────────

def send_mobile_otp(mobile: str, otp: str) -> Tuple[bool, str]:
    """
    Send OTP via SMS. Tries Twilio, falls back to terminal.
    """
    _print_otp("MOBILE SMS", f"+91{mobile}", otp)

    sid   = _env("TWILIO_SID")
    token = _env("TWILIO_TOKEN")
    from_ = _env("TWILIO_FROM")

    if not (sid and token and from_):
        print("[SMS] ⚠  Twilio not configured. OTP shown in terminal only.", flush=True)
        return True, "dev_console"

    try:
        import base64
        creds = base64.b64encode(f"{sid}:{token}".encode()).decode()
        body  = urllib.parse.urlencode({
            "From": from_,
            "To":   f"+91{mobile}",
            "Body": f"Your {APP_NAME} OTP: {otp}. Valid for 5 minutes. Do not share.",
        }).encode("utf-8")

        req = urllib.request.Request(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
            data=body,
            headers={
                "Authorization": f"Basic {creds}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = _json.loads(resp.read().decode())
            sid_msg = result.get("sid","")
            print(f"[TWILIO ✓] SMS sent to +91{mobile} | sid={sid_msg}", flush=True)
            return True, f"sent_twilio:{sid_msg}"

    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        print(f"[TWILIO ✗] HTTP {e.code}: {err[:200]}", flush=True)
        return False, f"Twilio HTTP {e.code}: {err[:100]}"

    except Exception as e:
        print(f"[TWILIO ✗] {type(e).__name__}: {e}", flush=True)
        return False, str(e)
