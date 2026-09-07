"""
Dashboard phân tích bóng đá (Streamlit).

Chạy:
  streamlit run app_dashboard.py
  hoặc ./run_dashboard.sh (Linux/Mac)
  hoặc run_dashboard.bat (Windows)

Tính năng chính:
- Hiển thị NGAY LẬP TỨC video kết quả phân tích & biểu đồ thống kê trận đấu khi vừa mở Dashboard.
- Hỗ trợ chạy mượt mà trên máy mới khi clone về (tự động cung cấp demo output, tự tải test.mp4 và best.pt từ GitHub Release nếu thiếu).
- Cho phép xem lại các lần chạy trước đó hoặc chạy phân tích video mới tùy chọn.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Cho phép chạy trực tiếp bằng cả 'python app_dashboard.py' lẫn 'streamlit run app_dashboard.py'
try:
    from streamlit import runtime
    if not runtime.exists():
        from streamlit.web import cli as stcli
        print("============================================================")
        print("⚽ Đang khởi động Football Match Analysis Web Dashboard...")
        print("👉 Vui lòng mở trình duyệt tại: http://localhost:8501")
        print("============================================================")
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())] + sys.argv[1:]
        sys.exit(stcli.main())
except (ImportError, AttributeError):
    pass

# ---- Khởi tạo đường dẫn dự án -------------------------------------------- #
BASE = Path(__file__).resolve().parent
FOOTBALL_DIR = BASE / "football"
DOWNLOADS_DIR = Path.home() / "Downloads"
REPORTS = BASE / "reports"
RUNS_DIR = REPORTS / "dash_runs"

# Đảm bảo các thư mục cần thiết luôn tồn tại
FOOTBALL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# URL tải dữ liệu mẫu từ GitHub Release (phòng khi máy khác clone về)
SAMPLE_VIDEO_URL = "https://github.com/ducanhdhtb06-hub/football-match-analysis/releases/download/v1.0.0/test.mp4"
SAMPLE_OUTPUT_URL = "https://github.com/ducanhdhtb06-hub/football-match-analysis/releases/download/v1.0.0/sample_output.mp4"


# ---- Các hàm tiện ích & Tải dữ liệu --------------------------------------- #
def ensure_sample_output() -> dict:
    """Đảm bảo có sẵn video kết quả mẫu và thống kê để hiển thị ngay khi mở Dashboard."""
    sample_video = REPORTS / "sample_output.mp4"
    if not sample_video.exists() or sample_video.stat().st_size == 0:
        try:
            print(f"[Dashboard] Đang tự động tải video output mẫu từ {SAMPLE_OUTPUT_URL}...")
            part = sample_video.with_suffix(".mp4.part")
            urllib.request.urlretrieve(SAMPLE_OUTPUT_URL, part)
            if part.exists() and part.stat().st_size > 0:
                part.rename(sample_video)
        except Exception as e:
            print(f"[Dashboard Warning] Không thể tải video output mẫu: {e}")

    sample_json = REPORTS / "sample_match_stats.json"
    sample_csv = REPORTS / "sample_match_stats_players.csv"
    sample_html = REPORTS / "sample_match_stats.html"

    return {
        "id": "sample_demo",
        "title": "📌 Video mẫu kết quả phân tích sẵn (Demo Output)",
        "video": sample_video if sample_video.exists() else None,
        "json": sample_json if sample_json.exists() else None,
        "csv": sample_csv if sample_csv.exists() else None,
        "html": sample_html if sample_html.exists() else None,
        "is_sample": True,
    }


def ensure_input_video() -> Path | None:
    """Tự động tải video đầu vào mẫu test.mp4 nếu trong project chưa có video nào."""
    target = FOOTBALL_DIR / "test.mp4"
    if not target.exists() or target.stat().st_size == 0:
        try:
            print(f"[Dashboard] Đang tải video đầu vào mẫu {SAMPLE_VIDEO_URL}...")
            part = target.with_suffix(".mp4.part")
            urllib.request.urlretrieve(SAMPLE_VIDEO_URL, part)
            if part.exists() and part.stat().st_size > 0:
                part.rename(target)
        except Exception as e:
            print(f"[Dashboard Warning] Không thể tải video mẫu test.mp4: {e}")
    return target if target.exists() else None


def find_videos() -> dict[str, list[Path]]:
    """Tìm tất cả các video đầu vào trong thư mục football/ và Downloads."""
    groups: dict[str, list[Path]] = {"Trong project (football/)": [], "Downloads": []}
    exts = (".mp4", ".webm", ".mov", ".mkv")
    if FOOTBALL_DIR.exists():
        groups["Trong project (football/)"] = sorted(
            p for p in FOOTBALL_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("output")
        )
    if DOWNLOADS_DIR.exists():
        groups["Downloads"] = sorted(
            p for p in DOWNLOADS_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("output")
        )

    # Nếu cả 2 đều trống, tự động đảm bảo có video mẫu test.mp4 trong football/
    if not groups["Trong project (football/)"] and not groups["Downloads"]:
        sample = ensure_input_video()
        if sample and sample.exists():
            groups["Trong project (football/)"] = [sample]

    return {k: v for k, v in groups.items() if v}


def list_available_runs() -> list[dict]:
    """Liệt kê các lần chạy kết quả trước đó và video mẫu demo."""
    runs: list[dict] = []
    if RUNS_DIR.exists():
        subdirs = sorted(
            [d for d in RUNS_DIR.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for d in subdirs:
            v = d / "result.mp4"
            if v.exists() and v.stat().st_size > 0:
                runs.append({
                    "id": d.name,
                    "title": f"🕒 Lần chạy: {d.name}",
                    "video": v,
                    "json": d / "match_stats.json" if (d / "match_stats.json").exists() else None,
                    "csv": d / "match_stats_players.csv" if (d / "match_stats_players.csv").exists() else None,
                    "html": d / "match_stats.html" if (d / "match_stats.html").exists() else None,
                    "is_sample": False,
                })

    # Luôn gắn bản mẫu demo sẵn có vào danh sách
    sample = ensure_sample_output()
    if sample["video"] is not None:
        runs.append(sample)
    return runs


def load_summary_json(path: Path | None):
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_analysis(video: Path, out_dir: Path, args: dict) -> Path:
    """Thực thi pipeline main.py và truyền stream log thời gian thực về giao diện."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    out_video = out_dir / "result.mp4"

    cmd = [
        sys.executable, "-u", str(BASE / "main.py"),
        "--video", str(video),
        "--output", str(out_video),
        "--stats-dir", str(out_dir),
        "--max-frames", str(args["max_frames"]),
        "--stride", str(args["stride"]),
        "--kp-every", str(args["kp_every"]),
    ]
    if args.get("stats"):
        cmd.append("--stats")
    if args.get("overlay"):
        cmd.append("--overlay")
    if args.get("voronoi"):
        cmd.append("--voronoi")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=log, stderr=subprocess.STDOUT, env=env)

    holder = st.empty()
    while True:
        time.sleep(0.4)
        if proc.poll() is not None:
            break
        try:
            new = log_path.read_text(encoding="utf-8", errors="ignore")
            holder.code(new[-4000:], language=None)
        except Exception:
            pass
    proc.wait()
    log.close()
    try:
        final = log_path.read_text(encoding="utf-8", errors="ignore")
        holder.code(final[-5000:], language=None)
    except Exception:
        pass

    if proc.returncode != 0:
        st.error(f"Chạy thất bại (exit code {proc.returncode}). Xem log phía trên để biết chi tiết.")
    return out_video


def render_results(run_data: dict):
    """Hiển thị toàn diện video kết quả phân tích, chỉ số, biểu đồ và bảng cầu thủ."""
    video_p = run_data.get("video")
    json_p = run_data.get("json")
    csv_p = run_data.get("csv")
    html_p = run_data.get("html")
    title = run_data.get("title", "Kết quả phân tích")
    is_sample = run_data.get("is_sample", False)

    if is_sample:
        st.info("🎬 **Video kết quả phân tích mẫu (Sẵn sàng ngay khi vào Dashboard).** Bạn có thể tải lên hoặc chọn video khác ở thanh menu bên trái và bấm **'Bắt đầu phân tích'** để xử lý video mới bất kỳ lúc nào.")
    else:
        st.success(f"🎬 **Đang hiển thị video kết quả phân tích: `{title}`**")

    # Thống kê tổng quan thẻ cards
    summary = load_summary_json(json_p)
    if summary:
        poss = summary.get("possession", {})
        share = poss.get("possession_share_pct", {})
        p = summary.get("processed", {})
        owner_frames = poss.get("owner_frames", 0)
        dur = float(p.get("duration_s", 0))

        def _s(team):
            return share.get(team, share.get(str(team), 0.0))

        col1, col2, col3, col4 = st.columns(4)
        if owner_frames > 0:
            col1.metric("Cầm bóng Đội 0 (Xanh)", f"{_s(0):.1f}%")
            col2.metric("Cầm bóng Đội 1 (Hồng)", f"{_s(1):.1f}%")
        else:
            col1.metric("Cầm bóng Đội 0", "—")
            col2.metric("Cầm bóng Đội 1", "—")
        col3.metric("Số frame phân tích", f"{p.get('frames', 0)}")
        col4.metric("Thời lượng trận", f"~{dur:.1f}s")

    # Khối thông tin vị trí lưu video kết quả
    if video_p and video_p.exists():
        try:
            rel_path = video_p.relative_to(BASE)
        except ValueError:
            rel_path = video_p
        v_size_mb = video_p.stat().st_size / (1024 * 1024)
        st.info(
            f"📁 **File video kết quả được lưu tại:** `{rel_path}` *({v_size_mb:.1f} MB)*  \n"
            f"👉 Bạn có thể mở trực tiếp file tại đường dẫn trên bằng trình phát video (VLC, Windows Media Player, QuickTime)."
        )
    else:
        st.warning("Chưa có video kết quả.")

    st.divider()

    # Bố cục 2 cột biểu đồ phân tích
    st.subheader("📊 Biểu đồ kiểm soát bóng & Diễn biến trận đấu")
    if summary:
        poss = summary.get("possession", {})
        share = poss.get("possession_share_pct", {})

        def _s(team):
            return share.get(team, share.get(str(team), 0.0))

        c1, c2 = st.columns(2)
        with c1:
            fig_donut = go.Figure(go.Pie(
                labels=["Đội 0 (Xanh)", "Đội 1 (Hồng)"],
                values=[_s(0), _s(1)],
                hole=0.45,
                marker=dict(colors=["#00BFFF", "#FF1493"])
            ))
            fig_donut.update_layout(
                title="Tỉ lệ kiểm soát bóng (%)",
                height=300,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            tl = poss.get("timeline_s", [])
            if tl:
                ca = cb = 0.0
                x, ya, yb = [], [], []
                for b in tl:
                    ca += b.get("a", 0)
                    cb += b.get("b", 0)
                    tot = max(ca + cb, 1e-6)
                    x.append(b["t"])
                    ya.append(100 * ca / tot)
                    yb.append(100 * cb / tot)
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=x, y=ya, name="Đội 0 (Xanh)", line=dict(color="#00BFFF", width=2)))
                fig_line.add_trace(go.Scatter(x=x, y=yb, name="Đội 1 (Hồng)", line=dict(color="#FF1493", width=2)))
                fig_line.update_layout(
                    title="Diễn biến kiểm soát bóng luỹ kế (%)",
                    height=300,
                    margin=dict(t=40, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Chưa có dữ liệu diễn biến thời gian.")
    else:
        st.info("Chưa có tệp dữ liệu thống kê match_stats.json.")

    # Bảng chi tiết từng cầu thủ
    if csv_p and csv_p.exists():
        st.divider()
        st.subheader("🏃‍♂️ Thống kê chi tiết từng cầu thủ (Vận tốc, Cự ly & Bứt tốc)")
        try:
            df_players = pd.read_csv(csv_p)
            st.dataframe(df_players, use_container_width=True)
        except Exception as e:
            st.warning(f"Không thể đọc bảng chi tiết cầu thủ: {e}")

    # Báo cáo đồ họa tương tác HTML
    if html_p and html_p.exists():
        st.divider()
        with open(html_p, "rb") as hf:
            st.download_button(
                label="📄 Tải toàn bộ báo cáo đồ họa tương tác HTML (match_stats.html)",
                data=hf,
                file_name="match_stats.html",
                mime="text/html",
                use_container_width=True,
            )


# ---- Cấu hình Trang Streamlit -------------------------------------------- #
st.set_page_config(page_title="⚽ Football Analysis Dashboard", page_icon="⚽", layout="wide")

st.title("⚽ Football Analysis & 2D Tactical Pitch Projection")
st.caption("Hệ thống AI phân tích chiến thuật, theo dõi cầu thủ, radar sân 2D và trích xuất chỉ số bóng đá tự động.")

# Đảm bảo danh sách các lần chạy
available_runs = list_available_runs()
if not available_runs:
    available_runs = [ensure_sample_output()]

# Mặc định hiển thị lần chạy mới nhất (hoặc bản mẫu demo khi mới clone)
if "active_run_id" not in st.session_state:
    st.session_state["active_run_id"] = available_runs[0]["id"]

# Đảm bảo active_run_id hợp lệ
valid_ids = [r["id"] for r in available_runs]
if st.session_state["active_run_id"] not in valid_ids:
    st.session_state["active_run_id"] = available_runs[0]["id"]

# ---- Thanh công cụ bên trái (Sidebar) ------------------------------------ #
with st.sidebar:
    st.header("📺 Chọn video kết quả xem")
    run_options = {r["id"]: r["title"] for r in available_runs}
    current_run_idx = valid_ids.index(st.session_state["active_run_id"])

    selected_run_id = st.selectbox(
        "Video kết quả đang hiển thị:",
        options=valid_ids,
        index=current_run_idx,
        format_func=lambda x: run_options[x],
        key="run_selector",
    )
    st.session_state["active_run_id"] = selected_run_id

    st.divider()
    st.header("🚀 Chạy phân tích video mới")

    # Tải lên video mới trực tiếp từ máy tính
    uploaded_file = st.file_uploader(
        "📁 Tải lên video mới (.mp4, .webm)",
        type=["mp4", "webm", "mov", "mkv"],
        help="Chọn video bóng đá từ máy của bạn để phân tích."
    )
    selected_video: Path | None = None
    if uploaded_file is not None:
        save_path = FOOTBALL_DIR / uploaded_file.name
        if not save_path.exists() or save_path.stat().st_size != uploaded_file.size:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        selected_video = save_path
        st.success(f"✅ Đã chọn video: `{uploaded_file.name}` ({uploaded_file.size / (1024 * 1024):.1f} MB)")
    else:
        st.caption("ℹ️ Kéo thả hoặc chọn video bóng đá ở trên để phân tích.")

    st.divider()
    max_frames = st.slider(
        "Số khung hình tối đa (Frames)", 30, 3000, 150, step=30,
        help="Giới hạn số frame để kiểm tra nhanh. 150 frames tương đương khoảng 5-6 giây."
    )
    full_video = st.checkbox("▶️ Phân tích TOÀN BỘ video", value=False)
    if full_video:
        max_frames = 999999

    stride = st.select_slider(
        "Stride (bước nhảy frame)", options=[1, 2, 3, 4], value=2,
        help="1 = xử lý từng frame; 2-3 = xử lý cách khung giúp tăng tốc 2-3 lần."
    )
    kp_every = st.slider(
        "Keypoints frequency (kp-every)", 1, 8, 3,
        help="Suy luận điểm mốc sân mỗi N frame để tăng tốc."
    )
    run_stats = st.checkbox("📊 Xuất báo cáo chỉ số (--stats)", value=True)
    run_overlay = st.checkbox("🎬 Live Overlay lên video (--overlay)", value=True)
    run_voronoi = st.checkbox("🔺 Vùng kiểm soát Voronoi (--voronoi)", value=False)

    start = st.button("🚀 Bắt đầu phân tích", type="primary", use_container_width=True, disabled=(selected_video is None))

# ---- Xử lý khi bấm 'Bắt đầu phân tích' ------------------------------------ #
if start and selected_video is not None:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in selected_video.stem)[:24]
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + safe
    out_dir = RUNS_DIR / stamp

    st.info(f"Đang tiến hành phân tích video: `{selected_video.name}`...")
    out_video = run_analysis(
        selected_video, out_dir,
        {
            "max_frames": max_frames,
            "stride": stride,
            "kp_every": kp_every,
            "stats": run_stats,
            "overlay": run_overlay,
            "voronoi": run_voronoi,
        }
    )

    if out_video.exists() and out_video.stat().st_size > 0:
        st.session_state["active_run_id"] = stamp
        st.success("🎉 Phân tích video hoàn tất!")
        st.rerun()
    else:
        st.error("Không tìm thấy file video kết quả sau khi chạy. Hãy kiểm tra lại log bên trên.")

# ---- HIỂN THỊ KẾT QUẢ VIDEO OUTPUT NGAY LẬP TỨC --------------------------- #
active_run_obj = next((r for r in available_runs if r["id"] == st.session_state["active_run_id"]), available_runs[0])
render_results(active_run_obj)

