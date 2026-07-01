@echo off
echo ========================================
echo   Setting up Virtual Environment
echo   My Data AI Project
echo ========================================

REM Check Python version
python --version

REM Create virtual environment named .venv
echo.
echo [1/4] Creating virtual environment (.venv)...
python -m venv .venv

REM Activate virtual environment
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

REM Install all requirements
echo [4/4] Installing project dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To activate your environment next time, run:
echo   .venv\Scripts\activate
echo.
echo To start the app, run:
echo   python run.py
echo.
pause
