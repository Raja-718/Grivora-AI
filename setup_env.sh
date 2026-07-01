# ========================================
#   Setup script for Linux / Mac users
# ========================================

echo "Creating virtual environment (.venv)..."
python3.11 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To activate next time:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the app:"
echo "  python run.py"
