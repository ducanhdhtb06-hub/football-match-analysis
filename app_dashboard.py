"""
Dashboard phân tích bóng đá (Streamlit).

Chạy:  streamlit run app_dashboard.py
- Chọn video (trong football/, Downloads hoặc tải lên từ máy tính)
- Tự động tải video mẫu test.mp4 từ GitHub Release nếu chưa có video
- Chọn tham số (số frame, stride, kp-every, bật stats/overlay/voronoi)
- Bấm "Chạy phân tích" -> theo dõi log -> xem video kết quả + biểu đồ ngay trên trang.
"""
from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

BASE = Path(__file__).resolve().parent
FOOTBALL_DIR = BASE / "football"
DOWNLOADS_DIR = Path.home() / "Downloads"
REPORTS = BASE / "reports"
RUNS_DIR = REPORTS / "dash_runs"

# Tạo các thư mục cần thiết
FOOTBALL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_VIDEO_URL = "https://github.com/ducanhdhtb06-hub/football-match-analysis/releases/download/v1.0.0/test.mp4"


def is_screen_recording(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("anh.webm", "nha", "ytdown", "vid1", "output", "demo"))


def find_videos() -> dict[str, list[Path]]:
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
    return {k: v for k, v in groups.items() if v}


def download_sample_video() -> Path:
    target = FOOTBALL_DIR / "test.mp4"
    urllib.request.urlretrieve(SAMPLE_VIDEO_URL, target)
    return target


def load_summary_json(path: Path):
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_analysis(video: Path, out_dir: Path, args: dict) -> Path:
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


def plot_summary(summary: dict):
    poss = summary.get("possession", {})
    share = poss.get("possession_share_pct", {})
    teams = summary.get("teams", {})
    p = summary.get("processed", {})
    owner_frames = poss.get("owner_frames", 0)
    dur = float(p.get("duration_s", 0))

    def _s(team):
        return share.get(team, share.get(str(team), 0.0))

    col1, col2, col3, col4 = st.columns(4)
    if owner_frames > 0:
        col1.metric("Cầm bóng Đội 0", f"{_s(0):.1f}%")
        col2.metric("Cầm bóng Đội 1", f"{_s(1):.1f}%")
    else:
        col1.metric("Cầm bóng Đội 0", "—")
        col2.metric("Cầm bóng Đội 1", "—")
    col3.metric("Số frames", f"{p.get('frames', 0)}")
    col4.metric("Thời lượng video", f"~{dur:.1f}s")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(
            labels=["Đội 0 (Xanh)", "Đội 1 (Hồng)"],
            values=[_s(0), _s(1)],
            hole=0.45,
            marker=dict(colors=["#00BFFF", "#FF1493"])
        ))
        fig.update_layout(title="Tỉ lệ kiểm soát bóng (%)", height=320)
        st.plotly_chart(fig, use_container_width=True)
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
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x, y=ya, name="Đội 0", line=dict(color="#00BFFF", width=2)))
            fig2.add_trace(go.Scatter(x=x, y=yb, name="Đội 1", line=dict(color="#FF1493", width=2)))
            fig2.update_layout(title="Diễn biến kiểm soát bóng luỹ kế (%)", height=320)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Chưa đủ dữ liệu timeline (video quá ngắn)")


# ---- Cấu hình Trang Streamlit -------------------------------------------- #
st.set_page_config(page_title="⚽ Football Analysis Dashboard", page_icon="⚽", layout="wide")

st.title("⚽ Football Analysis & 2D Tactical Pitch Projection")
st.caption("Phân tích chiến thuật, theo dõi cầu thủ, radar sân 2D và thống kê chỉ số bóng đá tự động.")

# ---- Tải danh sách video -------------------------------------------------- #
groups_full = find_videos()

with st.sidebar:
    st.header("⚙️ Cấu hình phân tích")

    # 1. Tải lên video mới
    uploaded_file = st.file_uploader(
        "📁 Tải lên video mới (.mp4, .webm)",
        type=["mp4", "webm", "mov", "mkv"],
        help="Chọn video bóng đá từ máy của bạn để thêm vào danh sách phân tích."
    )
    if uploaded_file is not None:
        save_path = FOOTBALL_DIR / uploaded_file.name
        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Đã lưu: `{uploaded_file.name}`")
            st.rerun()

    # 2. Tải video mẫu nếu chưa có
    if st.button("📥 Tải video mẫu test.mp4 (14MB)", help="Tải video mẫu trận đấu từ GitHub Release về máy"):
        with st.spinner("Đang tải video mẫu test.mp4..."):
            try:
                download_sample_video()
                st.success("Tải video mẫu thành công!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi tải video mẫu: {e}")

    # Cập nhật danh sách sau khi upload / download
    groups_full = find_videos()
    label_map = {}
    for group, lst in groups_full.items():
        for p in lst:
            label_map[f"{group} — {p.name}"] = p

    if label_map:
        ordered = list(label_map.keys())
        choice = st.selectbox(
            "Chọn video đầu vào",
            ordered,
            key="video_select",
        )
        selected_video = label_map[choice]
        st.caption(f"👉 File được chọn: `{selected_video.name}`")
    else:
        selected_video = None

    st.divider()
    max_frames = st.slider("Số khung hình tối đa (Frames)", 30, 3000, 150, step=30,
                           help="Giới hạn số frame để kiểm tra nhanh. 150 frames tương đương khoảng 5-6 giây.")
    full_video = st.checkbox("▶️ Phân tích TOÀN BỘ video", value=False)
    if full_video:
        max_frames = 999999

    stride = st.select_slider("Stride (bước nhảy frame)", options=[1, 2, 3, 4], value=2,
                              help="1 = xử lý từng frame; 2-3 = xử lý cách khung giúp tăng tốc 2-3 lần.")
    kp_every = st.slider("Keypoints frequency (kp-every)", 1, 8, 3,
                         help="Suy luận điểm mốc sân mỗi N frame để tăng tốc.")
    run_stats = st.checkbox("📊 Xuất báo cáo chỉ số (--stats)", value=True)
    run_overlay = st.checkbox("🎬 Live Overlay lên video (--overlay)", value=True)
    run_voronoi = st.checkbox("🔺 Vùng kiểm soát Voronoi (--voronoi)", value=False)

    start = st.button("🚀 Bắt đầu phân tích", type="primary", use_container_width=True, disabled=(selected_video is None))

# ---- Khung hiển thị chính ------------------------------------------------ #
if selected_video is None:
    st.info("👋 **Chưa có video nào trong hệ thống.**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Cách 1: Thử ngay với video mẫu")
        st.markdown("Bấm nút bên dưới để tự động tải clip trận đấu mẫu `test.mp4` từ GitHub:")
        if st.button("📥 Tải video mẫu ngay (14MB)", type="primary"):
            with st.spinner("Đang tải video mẫu..."):
                try:
                    download_sample_video()
                    st.success("Tải video mẫu thành công! Đang tải lại trang...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi tải video mẫu: {e}")
    with col_b:
        st.markdown("### Cách 2: Tải lên video trận đấu của bạn")
        st.markdown("Kéo thả hoặc duyệt file video bóng đá `.mp4` / `.webm` từ máy tính ở thanh menu bên trái.")
    st.stop()

st.subheader(f"Video đang chọn: `{selected_video.name}`")

if start:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in selected_video.stem)[:24]
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + safe
    out_dir = RUNS_DIR / stamp

    st.info(f"Đang phân tích video: `{selected_video.name}`...")
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
        st.success("🎉 Phân tích video hoàn tất!")

        # Phát video kết quả trực tiếp trên trang Web
        st.subheader("🎥 Video kết quả phân tích")
        try:
            st.video(str(out_video))
            with open(out_video, "rb") as vf:
                st.download_button(
                    label="💾 Tải video kết quả về máy (.mp4)",
                    data=vf,
                    file_name=f"analyzed_{selected_video.name}",
                    mime="video/mp4",
                )
        except Exception as e:
            st.warning(f"Không thể phát video trực tiếp trên trình duyệt: {e}. Bạn có thể mở file tại `{out_video}`.")

        # Hiển thị số liệu thống kê
        summary = load_summary_json(out_dir / "match_stats.json") if run_stats else None
        if summary:
            st.header("📊 Báo cáo chỉ số trận đấu")
            plot_summary(summary)

            # Bảng số liệu chi tiết cầu thủ nếu có
            csv_players = out_dir / "match_stats_players.csv"
            if csv_players.exists():
                try:
                    import pandas as pd
                    df_players = pd.read_csv(csv_players)
                    st.subheader("🏃‍♂️ Thống kê chi tiết từng cầu thủ")
                    st.dataframe(df_players, use_container_width=True)
                except Exception:
                    pass

            html_report = out_dir / "match_stats.html"
            if html_report.exists():
                with open(html_report, "rb") as hf:
                    st.download_button(
                        label="📄 Tải báo cáo đồ họa tương tác (match_stats.html)",
                        data=hf,
                        file_name="match_stats.html",
                        mime="text/html",
                    )
    else:
        st.error("Không tìm thấy file video kết quả sau khi chạy. Hãy kiểm tra lại log bên trên.")
