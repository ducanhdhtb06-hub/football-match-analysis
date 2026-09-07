"""
Dashboard phân tích bóng đá (Streamlit).

Chạy:  streamlit run app_dashboard.py
- Chọn video (trong football/ hoặc Downloads)
- Chọn tham số (số frame, stride, kp-every, bật stats/overlay/voronoi)
- Bấm "Chạy phân tích" -> theo dõi log -> xem kết quả + biểu đồ ngay trên trang.
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

BASE = Path(__file__).resolve().parent
FOOTBALL_DIR = BASE / "football"
DOWNLOADS_DIR = Path.home() / "Downloads"
REPORTS = BASE / "reports"
RUNS_DIR = REPORTS / "dash_runs"

# các thư mục chứa package (giống sys.path hack trong main.py)
EXTRA_PATHS = [
    Path.home() / "snap/antigravity-cli/common/local/lib/python3.12/dist-packages",
    Path.home() / "snap/antigravity-cli/common/lib/python3.12/site-packages",
]


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
            if p.is_file() and p.suffix.lower() in exts
        )
    return {k: v for k, v in groups.items() if v}


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
    if args["stats"]:
        cmd.append("--stats")
    if args["overlay"]:
        cmd.append("--overlay")
    if args["voronoi"]:
        cmd.append("--voronoi")

    env = dict(os.environ)
    extra = os.pathsep.join(str(p) for p in EXTRA_PATHS if p.exists())
    if extra:
        env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=log, stderr=subprocess.STDOUT, env=env)

    holder = st.empty()
    text = ""
    while True:
        time.sleep(0.4)
        if proc.poll() is not None:
            break
        try:
            new = log_path.read_text(encoding="utf-8", errors="ignore")
            text = new[-6000:]
            holder.code(text, language=None)
        except Exception:
            pass
    proc.wait()
    log.close()
    final = log_path.read_text(encoding="utf-8", errors="ignore")
    holder.code(final[-8000:], language=None)
    if proc.returncode != 0:
        st.error(f"Chạy thất bại (exit code {proc.returncode}). Xem log phía trên; "
                 f"thử giảm 'Số frame', tăng kp-every, hoặc kiểm tra network Roboflow.")
    return out_video


def plot_summary(summary: dict):
    poss = summary.get("possession", {})
    share = poss.get("possession_share_pct", {})
    teams = summary.get("teams", {})
    p = summary.get("processed", {})
    owner_frames = poss.get("owner_frames", 0)
    dur = float(p.get("duration_s", 0))
    def _s(team):
        # JSON load biến key số thành chuỗi "0"/"1" -> hỗ trợ cả 2
        return share.get(team, share.get(str(team), 0.0))

    col1, col2, col3, col4 = st.columns(4)
    if owner_frames > 0:
        col1.metric("Cầm bóng Đội 0", f"{_s(0):.1f}%")
        col2.metric("Cầm bóng Đội 1", f"{_s(1):.1f}%")
    else:
        col1.metric("Cầm bóng Đội 0", "—")
        col2.metric("Cầm bóng Đội 1", "—")
    col3.metric("Wobble toạ độ", f"{p.get('coordinate_wobble_est_ms', '?')} m/s")
    col4.metric("Chất lượng", str(p.get("coordinate_quality", "?")).split("(")[0])
    if owner_frames > 0:
        st.caption(f"Đã phân tích {p.get('frames', 0)} frame (~{dur:.1f}s). "
                   f"Gán được bóng {owner_frames} frame ({_s(0):.0f}% - {_s(1):.0f}%).")
    else:
        st.caption(f"Đã phân tích {p.get('frames', 0)} frame (~{dur:.1f}s). Chưa gán được bóng "
                   f"({owner_frames} frame) -> bóng không bám được ở video này.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(labels=["Đội 0", "Đội 1"],
                               values=[_s(0), _s(1)],
                               hole=0.45, marker=dict(colors=["#00BFFF", "#FF1493"])))
        fig.update_layout(title="Tỉ lệ cầm bóng", height=300)
        st.plotly_chart(fig, width='stretch')
    with c2:
        tl = poss.get("timeline_s", [])
        if tl:
            ca = cb = 0.0
            x, ya, yb = [], [], []
            for b in tl:
                ca += b.get("a", 0); cb += b.get("b", 0)
                tot = max(ca + cb, 1e-6)
                x.append(b["t"]); ya.append(100 * ca / tot); yb.append(100 * cb / tot)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x, y=ya, name="Đội 0", line=dict(color="#00BFFF")))
            fig2.add_trace(go.Scatter(x=x, y=yb, name="Đội 1", line=dict(color="#FF1493")))
            fig2.update_layout(title="Cầm bóng luỹ kế (%)", height=300)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Chưa đủ timeline (clip ngắn)")

    st.markdown("**Thống kê đội:**")
    st.json({k: v for k, v in (teams or {}).items()})


st.set_page_config(page_title="⚽ Phân tích bóng đá", page_icon="⚽", layout="wide")

# ---- tự tắt khi đóng hết tab trình duyệt -------------------------------- #
_watcher_started = False
def _start_auto_exit_watcher():
    global _watcher_started
    if _watcher_started:
        return
    _watcher_started = True
    import threading
    import time as _t
    import os as _os

    def _run():
        empty = 0
        while True:
            _t.sleep(10)
            n = 1
            try:
                from streamlit.runtime import get_instance as _gi
                mgr = _gi()._session_mgr
                n = len(mgr.list_active_sessions())
            except Exception:
                n = 1
            if n == 0:
                empty += 1
                if empty >= 6:      # ~60 giây không có tab nào
                    print("Khong con tab dashboard -> tu dong tat server.")
                    _os._exit(0)
            else:
                empty = 0

    threading.Thread(target=_run, daemon=True).start()

_start_auto_exit_watcher()

st.caption("Chọn video + tham số, bấm chạy. Kết quả nằm trong reports/dash_runs/<thời điểm>/")

# ---- load lại trang -> tự tắt phiên phân tích cũ còn chạy ngầm ------------- #
def _stop_previous_analysis():
    killed = []
    try:
        out = subprocess.run(["pgrep", "-f", "main.py --video"],
                             capture_output=True, text=True).stdout
        for pid in out.split():
            try:
                os.kill(int(pid), 9)
                killed.append(pid)
            except Exception:
                pass
    except Exception:
        pass
    lock = RUNS_DIR / ".running.lock"
    try:
        if lock.exists():
            lock.unlink()
    except Exception:
        pass
    return killed

_killed = _stop_previous_analysis()
if _killed:
    st.info(f"Đã tự dừng {len(_killed)} phiên phân tích cũ khi tải lại trang.")

groups_full = find_videos()
with st.sidebar:
    st.header("⚙️ Cấu hình")
    if not groups_full:
        st.error("Không tìm thấy video!")
        st.stop()
    hide_screen = st.checkbox("Ẩn clip quay màn hình YouTube (khuyên bật)", value=True)
    groups = {}
    for g, lst in groups_full.items():
        filtered = [p for p in lst if not (hide_screen and is_screen_recording(p.name))]
        if filtered:
            groups[g] = filtered
    if not any(groups.values()):
        groups = groups_full

    label_map = {}
    for group, lst in groups.items():
        for p in lst:
            label_map[f"{group} — {p.name}"] = p

    ordered = list(label_map.keys())
    choice = st.selectbox(
        "Video đầu vào",
        ordered,
        key="video_select",
        help="Chọn file nào thì chạy đúng file đó.",
    )
    video = label_map[choice]
    st.caption(f"👉 Sẽ chạy: `{video}`")

    max_frames = st.slider("Số frame tối đa", 30, 9000, 300, step=30,
                           help="Giới hạn số frame xử lý (xem mốc thời gian để ước tính thời gian chạy).")
    full_video = st.checkbox("▶️ Xử lý TOÀN BỘ video (bỏ giới hạn frame)", value=False)
    if full_video:
        max_frames = 999999
        st.caption("Đã chọn toàn bộ video — có thể rất lâu với clip dài (xem gợi ý stride/kp-every bên dưới).")
    stride = st.select_slider("Stride (1 = mọi frame, 2-4 = nhanh hơn cho clip dài)",
                              options=[1, 2, 3, 4], value=1)
    kp_every = st.slider("kp-every (gọi keypoint cách N frame, nhanh hơn)", 1, 8, 3)
    run_stats = st.checkbox("📊 Xuất báo cáo chỉ số (--stats)", value=True)
    run_overlay = st.checkbox("🎬 Overlay chỉ số lên video (--overlay)", value=True)
    run_voronoi = st.checkbox("🔺 Voronoi trên radar (--voronoi)", value=False)
    start = st.button("🚀 Chạy phân tích", type="primary", use_container_width=True)

st.subheader("Video đã chọn")
st.text(f"{video.name}  ·  {video.parent}")

if start:
    # ---- khoá chống chạy trùng (nhiều tab / bấm nhiều lần) ---------------- #
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    lock = RUNS_DIR / ".running.lock"
    if lock.exists():
        import time as _t
        import subprocess as _sp
        stale = _t.time() - lock.stat().st_mtime > 20
        if stale:
            # tự dọn nếu KHÔNG còn tiến trình phân tích thật nào
            alive = _sp.run(["pgrep", "-f", "main.py --video"],
                            capture_output=True).returncode == 0
            if not alive:
                lock.unlink()
    if lock.exists():
        st.error("⚠️ Đang có MỘT phân tích khác chạy (bạn mở 2 dashboard hay bấm Run nhiều lần?). "
                 "Hãy chờ run kia xong, hoặc xoá file lock: reports/dash_runs/.running.lock rồi bấm lại.")
        st.stop()
    lock.write_text(str(os.getpid()), encoding="utf-8")

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in video.stem)[:24]
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + safe
    out_dir = RUNS_DIR / stamp
    st.info(f"Đang chạy: `{video.name}` → {out_dir}")
    out_video = run_analysis(
        video, out_dir,
        {"max_frames": max_frames, "stride": stride, "kp_every": kp_every,
         "stats": run_stats, "overlay": run_overlay, "voronoi": run_voronoi},
    )
    st.success("Hoàn tất!")
    st.markdown(f"**Video kết quả:** `{out_video}`")
    st.markdown(f"**Video đã chạy:** `{video.name}`")
    st.caption("Nếu cầm bóng 0%: kiểm tra tên video trên — clip quay màn hình (anh/nha/ytdown) sẽ ra 0%.")

    summary = load_summary_json(out_dir / "match_stats.json") if run_stats else None
    if summary:
        st.header("📊 Kết quả phân tích")
        plot_summary(summary)
        html = out_dir / "match_stats.html"
        xlsx = out_dir / "match_stats.xlsx"
        st.markdown("**Báo cáo (mở bằng PyCharm/trình duyệt):**")
        for f in (html, xlsx, out_dir / "match_stats.json"):
            if f.exists():
                st.code(str(f))
        st.caption("Tips: mở match_stats.html bằng trình duyệt, .xlsx bằng Excel/LibreOffice.")
    else:
        st.info("Không có báo cáo stats (bạn tắt --stats?) — chỉ có video kết quả.")
    try:
        lock.unlink()
    except Exception:
        pass
