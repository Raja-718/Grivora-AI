"""
STARTUP CHECKLIST — run this to verify routes.py parses without errors.
Usage: python check_syntax.py
"""
import py_compile, sys

files = [
    'app/routes.py',
    'app/auto_dashboard_route.py',
    'auth_system/auth_middleware.py',
]

ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f'OK  {f}')
    except py_compile.PyCompileError as e:
        print(f'ERR {f}: {e}')
        ok = False

sys.exit(0 if ok else 1)
