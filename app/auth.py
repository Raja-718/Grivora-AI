# auth.py - Login / session handling
from flask import session, redirect, url_for, request

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated
