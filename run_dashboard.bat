@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   Football Match Analysis - Khoi dong Web Dashboard
echo ============================================================

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

where streamlit >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    streamlit run app_dashboard.py %*
    goto end
)

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app_dashboard.py %*
    goto end
)

if exist "venv\Scripts\streamlit.exe" (
    "venv\Scripts\streamlit.exe" run app_dashboard.py %*
    goto end
)

python -m streamlit --version >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python -m streamlit run app_dashboard.py %*
    goto end
)

echo.
echo ============================================================
echo [Loi] Chua tim thay Streamlit hoac thu vien can thiet!
echo ============================================================
echo Vui long mo CMD / Terminal trong thu muc nay va chay:
echo   python -m venv .venv
echo   .venv\Scripts\activate
echo   pip install -r requirements.txt
echo.
pause

:end
