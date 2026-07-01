@echo off
echo ══════════════════════════════════════════════════════
echo  Omniora AI — Installing Universal Data Support Packages
echo ══════════════════════════════════════════════════════
echo.

echo [1/8] Installing core packages...
pip install pandas>=2.0.0 numpy>=1.24.0

echo.
echo [2/8] Installing Excel support (xlsx, xls, xlsb, ods)...
pip install "openpyxl>=3.1.0" "xlrd>=2.0.1" "pyxlsb>=1.0.10" "odfpy>=1.4.1"

echo.
echo [3/8] Installing encoding auto-detection (chardet)...
pip install "chardet>=5.0.0"

echo.
echo [4/8] Installing Parquet support...
pip install "pyarrow>=12.0.0"

echo.
echo [5/8] Installing ML packages...
pip install "scikit-learn>=1.3.0"

echo.
echo [6/8] Installing database drivers...
pip install "mysql-connector-python>=8.0.0" "sqlalchemy>=2.0.0"

echo.
echo [7/8] Installing AI packages...
pip install "google-generativeai>=0.5.0" "python-dotenv>=1.0.0"

echo.
echo [8/8] Installing report generation...
pip install "reportlab>=4.0.0"

echo.
echo ══════════════════════════════════════════════════════
echo  ✅ All packages installed successfully!
echo  Now restart the server: python run.py
echo ══════════════════════════════════════════════════════
pause
