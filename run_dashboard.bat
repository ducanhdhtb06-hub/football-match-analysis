@echo off
cd /d "%~dp0"

echo ===================================================
echo   Football Match Analysis - Khoi dong Web Dashboard
echo ===================================================

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app_dashboard.py %*
) else (
    streamlit run app_dashboard.py %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Loi] Chua cai dat Streamlit hoac thu vien can thiet!
    echo Vui long chay: pip install -r requirements.txt
    pause
)
