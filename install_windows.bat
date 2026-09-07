@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   Football Match Analysis - Cai dat moi truong (Windows)
echo ============================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [Loi] May chua cai dat Python!
    echo Vui long truy cap https://www.python.org/ va cai dat Python 3.10 tro len.
    echo LUU Y: Khi cai dat, nho tich vao o "Add Python to PATH"!
    echo.
    pause
    exit /b 1
)

echo [1/3] Dang khoi tao moi truong ao (.venv)...
if not exist ".venv" (
    python -m venv .venv
    echo Da tao thu muc .venv thanh cong!
) else (
    echo Moi truong ao .venv da ton tai san.
)

echo.
echo [2/3] Dang kich hoat moi truong ao va cap nhat pip...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip

echo.
echo [3/3] Dang cai dat cac thu vien (PyTorch, YOLO, Streamlit, OpenCV...)...
echo Qua trinh nay co the mat tu 3 den 10 phut tuy thuoc vao toc do mang.
pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo [Thanh cong] Cai dat moi truong hoan tat 100%!
    echo ============================================================
    echo Tu bay gio, ban chi can nhap dup chuot vao file "run_dashboard.bat"
    echo de mo Web Dashboard bat cu luc nao.
    echo.
    set /p LAUNCH="Ban co muon mo Web Dashboard ngay bay gio luon khong? (Y/N): "
    if /i "!LAUNCH!"=="Y" (
        streamlit run app_dashboard.py
    )
) else (
    echo.
    echo [Canh bao] Co loi xay ra trong qua trinh cai dat thu vien.
    echo Vui long kiem tra ket noi mang va thu chay lai file nay.
)

pause
