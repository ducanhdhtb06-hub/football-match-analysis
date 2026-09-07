#!/usr/bin/env bash
# Dashboard tự tắt ~60s sau khi bạn đóng tab/out khỏi dashboard.
cd "$(dirname "$0")"

# dừng server cũ nếu có (tránh đè cổng)
ps aux | grep '[s]treamlit run app_dashboard' | awk '{print $2}' | xargs -r kill 2>/dev/null
sleep 1

.venv/bin/streamlit run app_dashboard.py --server.headless true --server.port 8501 > dash_server.log 2>&1 &
SPID=$!

echo "======================================================"
echo "  Dashboard:  http://localhost:8501"
echo "  (Tự động tắt sau ~60s kể từ khi bạn đóng tab.)"
echo "  Để tắt ngay: Ctrl+C ở terminal này."
echo "======================================================"

empty=0
while kill -0 $SPID 2>/dev/null; do
  if command -v ss >/dev/null 2>&1; then
    CONN=$(ss -tn 2>/dev/null | grep -c ':8501')
  else
    CONN=1
  fi
  if [ "${CONN:-0}" -eq 0 ]; then empty=$((empty+1)); else empty=0; fi
  if [ "$empty" -ge 12 ]; then   # ~60 giây không ai kết nối
    echo "Không còn ai mở dashboard -> tự tắt."
    kill $SPID 2>/dev/null
    break
  fi
  sleep 5
done
wait $SPID 2>/dev/null
echo "Đã tắt dashboard."
