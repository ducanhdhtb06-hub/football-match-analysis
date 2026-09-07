#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ -f ".venv/bin/streamlit" ]; then
    exec .venv/bin/streamlit run app_dashboard.py "$@"
elif command -v streamlit >/dev/null 2>&1; then
    exec streamlit run app_dashboard.py "$@"
else
    echo "Lỗi: Không tìm thấy Streamlit!"
    echo "Hãy kích hoạt môi trường ảo hoặc cài đặt thư viện trước:"
    echo "  pip install -r requirements.txt"
    exit 1
fi
