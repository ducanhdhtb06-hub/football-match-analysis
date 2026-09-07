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
echo [Thong bao] Chua tim thay moi truong ao hoac Streamlit!
echo ============================================================
echo.
set /p AUTO_INSTALL="Ban co muon he thong TU DONG cai dat thu vien ngay bay gio khong? (Y/N): "
if /i "!AUTO_INSTALL!"=="Y" (
    echo.
    echo [1/3] Dang kiem tra Python...
    where python >nul 2>nul
    if !ERRORLEVEL! NEQ 0 (
        echo [Loi] May chua cai dat Python! Vui long cai dat Python tai https://www.python.org/ (nho tich "Add Python to PATH").
        pause
        goto end
    )
    echo [2/3] Dang khoi tao moi truong ao (.venv)...
    python -m venv .venv
    call ".venv\Scripts\activate.bat"
    echo [3/3] Dang cai dat cac thu vien can thiet (co the mat vai phut tuy toc do mang)...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo.
    echo ============================================================
    echo [Thanh cong] Cai dat hoan tat! Dang mo Web Dashboard...
    echo ============================================================
    streamlit run app_dashboard.py
    goto end
)

echo.
echo Hoac ban co the tu cai dat bang tay:
echo   python -m venv .venv
echo   .venv\Scripts\activate
echo   pip install -r requirements.txt
echo   run_dashboard.bat
echo.
pause

:end
