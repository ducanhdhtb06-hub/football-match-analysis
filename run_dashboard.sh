#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Run Streamlit
if [ -f ".venv/bin/streamlit" ]; then
    exec .venv/bin/streamlit run app_dashboard.py "$@"
elif [ -f "venv/bin/streamlit" ]; then
    exec venv/bin/streamlit run app_dashboard.py "$@"
elif command -v streamlit >/dev/null 2>&1; then
    exec streamlit run app_dashboard.py "$@"
elif command -v python3 >/dev/null 2>&1 && python3 -m streamlit --version >/dev/null 2>&1; then
    exec python3 -m streamlit run app_dashboard.py "$@"
elif command -v python >/dev/null 2>&1 && python -m streamlit --version >/dev/null 2>&1; then
    exec python -m streamlit run app_dashboard.py "$@"
else
    echo "============================================================"
    echo "Lỗi: Không tìm thấy Streamlit!"
    echo "============================================================"
    echo "Hãy cài đặt thư viện trước khi chạy:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo "Sau đó chạy lại: ./run_dashboard.sh"
    exit 1
fi
