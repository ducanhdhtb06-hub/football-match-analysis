"""
Match statistics engine for football analysis videos.

Collects per-frame tracking + pitch-projection data produced by the pipeline
and computes a set of football metrics after the video has been processed:

    - Ball possession (% per team, neutral time, longest control runs)
    - Sprint / speed statistics (top speed km/h, number of sprints, avg speed)
    - Distance covered (per player and per team)
    - Average position + activity heat-map per team on the 2D pitch

All spatial data uses the pipeline's pitch frame: x = length axis (cm,
0..pitch.length), y = width axis (cm, 0..pitch.width) with the origin at the
top-left corner of the template pitch. Results are exported as JSON + CSV +
a self-contained interactive HTML report (Plotly).

IMPORTANT — heuristics & limitations:
  * Possession is *estimated* from geometry: at each frame the ball is
    credited to the team whose nearest out-field player is within
    ``possession_radius_cm`` of the ball. When the ball is not detected the
    previous owner is kept for a short ``carry_over_s`` window.
  * Speeds are derived from consecutive projected positions, so their accuracy
    depends on detection/homography stability and on a *true* video FPS.
    Jumps that imply more than ``max_speed_kmh`` are clipped.
  * Sprint = sustained smoothed speed >= ``sprint_kmh`` for at least
    ``sprint_min_duration_s``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pitch_config import SoccerPitchConfiguration

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class MatchStatsConfig:
    """Tunable thresholds for the statistics computation."""

    # ---- possession ------------------------------------------------------- #
    possession_radius_cm: float = 300.0      # max distance ball->player to own the ball
    carry_over_s: float = 0.7                # keep previous owner after ball gets loose/unseen
    loose_break_s: float = 2.5               # longer loose period breaks a possession run

    # ---- speed / sprint --------------------------------------------------- #
    max_speed_kmh: float = 43.2              # 12 m/s, physical plausibility cap
    sprint_kmh: float = 25.0                 # sprint threshold (~7 m/s)
    sprint_min_duration_s: float = 0.5       # a sprint must last at least this long
    smoothing_window: int = 9                # robust median smoothing of the speed series
    min_track_obs: int = 10                  # ignore tracks with fewer observations
    track_gap_s: float = 1.0                 # larger dt splits a track into segments

    # ---- robust estimation under detection/homography noise -------------- #
    jump_gate_m: float = 2.5                 # larger per-step jump => glitch (dropped)
    speed_window_s: float = 0.5              # (legacy window) kept for compatibility
    distance_min_speed_ms: float = 0.5       # slower apparent movement = standing (noise)

    # ---- per-player velocity filter (alpha-beta / constant velocity) ----- #
    filter_alpha: float = 0.35               # position gain (smoother)
    filter_beta: float = 0.18                # velocity gain (lower = smoother)

    # ---- short track re-identification (merge broken IDs) ---------------- #
    merge_max_gap_s: float = 2.0             # candidate must disappear this long max
    merge_max_dist_m: float = 4.0            # and reappear near the old position

    # ---- flicker-pair merge (2 IDs cua CUNG 1 nguoi xuat hien xen ke) ---- #
    flicker_merge_dist_m: float = 2.5        # median distance khi thay nhau
    flicker_min_samples: int = 10            # so lan kiem tra toi thieu
    flicker_time_window_s: float = 0.3       # chi so sanh khi cach nhau <= window

    # ---- spatial filtering ------------------------------------------------ #
    out_of_bounds_margin_cm: float = 600.0   # allow positions slightly outside lines
    heatmap_cell_cm: float = 200.0           # 2 m x 2 m cells

    # ---- output ----------------------------------------------------------- #
    report_basename: str = "match_stats"


@dataclass
class FrameRecord:
    """One processed frame captured by the recorder."""
    t: float                                  # seconds (real video time)
    ok: bool                                  # homography/projection valid this frame
    ball_xy: Optional[Tuple[float, float]]    # ball pitch coords in cm (or None)
    players: List[Tuple[int, int, float, float]] = field(default_factory=list)
    # players entries: (tracker_id, team_id 0/1, x_cm, y_cm)


class MatchStatsRecorder:
    """Lightweight collector appended once per processed frame."""

    def __init__(self, config: Optional[MatchStatsConfig] = None):
        self.config = config or MatchStatsConfig()
        self._frames: List[FrameRecord] = []
        self._count = 0

    def record(
        self,
        t_sec: float,
        ball_xy: Optional[Tuple[float, float]],
        players: Sequence[Tuple[int, int, float, float]],
        ok: bool,
    ) -> None:
        """Append one processed frame (called by the pipeline)."""
        self._frames.append(
            FrameRecord(t=t_sec, ok=bool(ok), ball_xy=ball_xy,
                        players=[tuple(p) for p in players])
        )
        self._count += 1

    def __len__(self) -> int:
        return self._count

    @property
    def frames(self) -> List[FrameRecord]:
        return self._frames


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


@dataclass
class PlayerStatsRow:
    tracker_id: int
    team: int
    seconds_active: float
    n_obs: int
    distance_m: float = 0.0
    avg_speed_kmh: float = 0.0
    top_speed_kmh: float = 0.0
    sprint_count: int = 0
    sprint_distance_m: float = 0.0
    avg_x_m: float = 0.0
    avg_y_m: float = 0.0
    speed_reliable: bool = True   # False = ID flicker/đo không tin cậy


class MatchStatsAnalyzer:
    """Turn raw FrameRecords into a statistics summary."""

    def __init__(
        self,
        frames: Sequence[FrameRecord],
        pitch_config: SoccerPitchConfiguration,
        config: Optional[MatchStatsConfig] = None,
        video_meta: Optional[dict] = None,
    ):
        self.frames = list(frames)
        self.pitch = pitch_config
        self.cfg = config or MatchStatsConfig()
        self.video_meta = video_meta or {}

        self.length = float(self.pitch.length)     # cm
        self.width = float(self.pitch.width)       # cm
        self.margin = self.cfg.out_of_bounds_margin_cm

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _in_bounds(self, x: float, y: float) -> bool:
        return (
            -self.margin <= x <= self.length + self.margin
            and -self.margin <= y <= self.width + self.margin
        )

    def _clamp(self, x: float, y: float) -> Tuple[float, float]:
        return (
            min(max(x, 0.0), self.length),
            min(max(y, 0.0), self.width),
        )

    # ------------------------------------------------------------------ #
    # players: distance / speed / sprints
    # ------------------------------------------------------------------ #

    def _analyze_track(self, tid: int, obs: List[Tuple[float, float, float]]) -> PlayerStatsRow:
        """obs: sorted list of (t_sec, x_cm, y_cm) for one tracker id."""
        cfg = self.cfg
        n = len(obs)
        t0, t1 = obs[0][0], obs[-1][0]
        row = PlayerStatsRow(tracker_id=tid, team=-1, seconds_active=max(0.0, t1 - t0), n_obs=n)
        if n < cfg.min_track_obs:
            return row

        ts = np.array([o[0] for o in obs])
        xs = np.array([o[1] for o in obs], dtype=np.float64)
        ys = np.array([o[2] for o in obs], dtype=np.float64)

        # split into contiguous segments separated by large gaps
        dts = np.diff(ts)
        split_at = np.where(dts > cfg.track_gap_s)[0] + 1
        segs = np.split(np.arange(n), split_at)

        dist_m = 0.0
        v_kmh: List[float] = []          # filtered speed per accepted update (km/h)
        step_d_m: List[float] = []       # per-update distance contribution (m)
        dt_used: List[float] = []        # dt of each accepted update (s)
        cap_ms = cfg.max_speed_kmh / 3.6
        min_ms = cfg.distance_min_speed_ms
        alpha = cfg.filter_alpha
        beta = cfg.filter_beta

        for seg in segs:
            if len(seg) < 3:
                continue
            st = ts[seg]
            raw = np.stack([xs[seg], ys[seg]], axis=1)

            # ---- outlier gate ------------------------------------------- #
            # Drop *isolated* spikes only: a point is rejected when it jumps
            # more than jump_gate_m from BOTH the previous and the next point.
            # Persistent (real) movement or a re-based homography is kept.
            gate_cm = cfg.jump_gate_m * 100.0
            keep = np.ones(len(raw), dtype=bool)
            for j in range(1, len(raw)):
                d_prev = math.hypot(raw[j, 0] - raw[j - 1, 0], raw[j, 1] - raw[j - 1, 1])
                if d_prev > gate_cm:
                    if j + 1 < len(raw):
                        d_next = math.hypot(raw[j + 1, 0] - raw[j, 0],
                                            raw[j + 1, 1] - raw[j, 1])
                        if d_next > gate_cm:
                            keep[j] = False
                    else:
                        keep[j] = False  # dangling last point after a jump
            keep = np.where(keep)[0]
            if len(keep) < 3:
                continue
            sa = raw[keep]
            ta = st[keep]

            # ---- alpha-beta (constant-velocity) filter ------------------- #
            # State: position (cm) + velocity (cm/s).  Velocity only changes
            # through innovations that pass the outlier gate, so the resulting
            # speed series has far less detection/homography noise than a raw
            # finite difference and stays responsive to real acceleration.
            px, py = float(sa[0, 0]), float(sa[0, 1])
            # bootstrap velocity from the first two accepted samples so a new
            # segment does not start from stand-still
            if len(keep) >= 2:
                dt0 = ta[1] - ta[0]
                if dt0 > 1e-6:
                    vx = (float(sa[1, 0]) - px) / dt0
                    vy = (float(sa[1, 1]) - py) / dt0
                    start_j = 1
                else:
                    vx = vy = 0.0
                    start_j = 0
            else:
                vx = vy = 0.0
                start_j = 0
            pt = ta[start_j]
            for j in range(start_j + 1, len(keep)):
                dtj = ta[j] - pt
                if dtj <= 1e-6:
                    continue
                # predict
                px += vx * dtj
                py += vy * dtj
                ex = float(sa[j, 0]) - px
                ey = float(sa[j, 1]) - py
                if math.hypot(ex, ey) > gate_cm:
                    continue            # innovation outlier: keep prediction only
                # correct
                px += alpha * ex
                py += alpha * ey
                vx += (beta / dtj) * ex
                vy += (beta / dtj) * ey
                v_ms = min(math.hypot(vx, vy) / 100.0, cap_ms)   # m/s
                v_kmh.append(v_ms * 3.6)
                dt_used.append(dtj)
                step = v_ms * dtj if v_ms >= min_ms else 0.0
                step_d_m.append(step)
                dist_m += step
                pt = ta[j]

        if not v_kmh:
            return row

        # ---- final smoothing + aggregate stats --------------------------- #
        # Robust median filter on the speed series: unlike a moving mean it
        # removes single-frame glitch spikes without biasing real acceleration.
        arr = np.array(v_kmh, dtype=np.float64)
        w = max(1, cfg.smoothing_window)
        if w % 2 == 0:
            w += 1
        if len(arr) >= w:
            pad = w // 2
            ext = np.concatenate([
                np.full(pad, arr[0]), arr, np.full(pad, arr[-1])
            ])
            smoothed = np.array(
                [float(np.median(ext[i:i + w])) for i in range(len(arr))]
            )
        else:
            smoothed = arr.copy()
        smoothed = np.maximum(smoothed, 0.0)
        steps = np.array(step_d_m, dtype=np.float64)

        row.distance_m = round(dist_m, 2)
        moving = smoothed > 0.6  # km/h, ignore residual noise when averaging
        row.avg_speed_kmh = round(float(smoothed[moving].mean()), 2) if moving.any() else 0.0
        # top speed = robust p95 of the smoothed series; tracks that are too
        # short carry no reliable top speed (0 = not enough data)
        if len(smoothed) >= 20:
            row.top_speed_kmh = round(float(np.percentile(smoothed, 95.0)), 2)
        else:
            row.top_speed_kmh = 0.0
        # reliability: nếu p95 > 36 km/h (trên mức con người) => dính trần/flicker
        row.speed_reliable = bool(row.top_speed_kmh <= 36.0)

        # ---- sprint detection ------------------------------------------- #
        # A sprint = contiguous filtered speeds >= sprint_kmh.
        over = smoothed >= cfg.sprint_kmh
        if over.any():
            pos_dts = np.array([d for d in dt_used if d > 1e-6])
            dt_sample = float(np.median(pos_dts)) if len(pos_dts) else 1.0 / 30.0
            run_start = None
            runs = []
            for i, flag in enumerate(over):
                if flag and run_start is None:
                    run_start = i
                elif not flag and run_start is not None:
                    runs.append((run_start, i - 1))
                    run_start = None
            if run_start is not None:
                runs.append((run_start, len(over) - 1))

            for s, e in runs:
                dur = (e - s + 1) * dt_sample
                seg_dist = float(steps[s:e + 1].sum())
                if dur >= max(cfg.sprint_min_duration_s, 0.6) and seg_dist >= 3.0:
                    row.sprint_count += 1
                    row.sprint_distance_m += seg_dist

        row.sprint_distance_m = round(row.sprint_distance_m, 2)

        # average position
        valid_mask = self._valid_position_mask(xs, ys)
        if valid_mask.any():
            cx, cy = self._clamp(float(xs[valid_mask].mean()), float(ys[valid_mask].mean()))
            row.avg_x_m = round(cx / 100.0, 2)
            row.avg_y_m = round(cy / 100.0, 2)

        # team is resolved by caller (map over track id per frame) ----------
        return row

    def _valid_position_mask(self, xs, ys) -> np.ndarray:
        m = (
            (xs >= -self.margin) & (xs <= self.length + self.margin)
            & (ys >= -self.margin) & (ys <= self.width + self.margin)
        )
        return m

    def _estimate_coordinate_quality(self) -> Tuple[float, str]:
        """Rough self-diagnosis of projection noise (m/s of shared wobble)."""
        drift_cm: List[float] = []
        prev: Dict[int, Tuple[float, float]] = {}
        for fr in self.frames:
            cur = {
                tid: (x, y)
                for tid, team, x, y in fr.players
                if team in (0, 1) and self._in_bounds(x, y)
            }
            common = cur.keys() & prev.keys()
            if len(common) >= 3:
                ds = [math.hypot(cur[t][0] - prev[t][0], cur[t][1] - prev[t][1])
                      for t in common]
                drift_cm.append(float(np.median(ds)))
            prev = cur

        ts = np.array([f.t for f in self.frames])
        dts = np.diff(ts)
        dts = dts[dts > 1e-6]
        dt_med = float(np.median(dts)) if len(dts) else 1.0 / 30.0
        wobble_ms = (float(np.median(drift_cm)) / 100.0) / dt_med if drift_cm else 0.0

        if wobble_ms < 1.2:
            quality = "tốt (homography ổn định)"
        elif wobble_ms < 4.0:
            quality = "trung bình"
        else:
            quality = "kém (nhiễu homography/detection mạnh — chỉ số nên xem là ước lượng thô)"
        return round(wobble_ms, 2), quality

    # ------------------------------------------------------------------ #
    # possession
    # ------------------------------------------------------------------ #

    def _analyze_possession(self, team_of: Dict[int, int]) -> dict:
        cfg = self.cfg
        stat_frames = 0            # frames usable for possession (ok & ball seen)
        owner_frames = 0           # frames with an effective owner (0/1)
        owner_counts = {0: 0, 1: 0}
        last_owner: Optional[int] = None
        last_resolved_t: Optional[float] = None
        # per-player control (relative distances -> robust to homography wobble)
        control: Dict[int, Dict[str, float]] = {}   # tid -> control_s, touches
        prev_res_tid: Optional[int] = None
        prev_res_t: Optional[float] = None
        # run bookkeeping
        run_start_t: Optional[float] = None
        run_owner: Optional[int] = None
        runs: List[Tuple[int, float, float]] = []      # (team, start_t, end_t)
        neutral_s = 0.0
        prev_t = None
        prev_ok_t: Optional[float] = None
        # second-by-second possession timeline (bins by whole second)
        timeline: Dict[int, List[float]] = {}          # sec -> [team0_s, team1_s, neutral_s]
        # ball zone (1/3 thirds along pitch length) per owning team
        zone_s = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]}
        third = self.length / 3.0

        # ---- ball trajectory with short-gap linear interpolation --------- #
        # If the ball disappears for <=0.5 s it is interpolated between the two
        # surrounding detections; longer gaps stay unknown.  This keeps the
        # possession/timeline continuous across brief occlusions/detection misses.
        ok_frames = [f for f in self.frames if f.ok]
        ball_seq: List[Optional[Tuple[float, float]]] = []
        for f in ok_frames:
            if f.ball_xy is not None:
                bx0, by0 = float(f.ball_xy[0]), float(f.ball_xy[1])
                if math.isfinite(bx0) and math.isfinite(by0):
                    ball_seq.append((bx0, by0))
                    continue
            ball_seq.append(None)

        # (không nội suy: giữ đúng luật realtime của thanh video để khớp 100%)

        for idx, fr in enumerate(ok_frames):
            ball_pos = ball_seq[idx]
            if ball_pos is None:
                continue                    # long unknown stretch: skip
            bx, by = ball_pos
            stat_frames += 1

            # nearest player to the ball (id + team)
            nearest_team: Optional[int] = None
            nearest_tid: Optional[int] = None
            best_d = float("inf")
            for tid, team, x, y in fr.players:
                if team not in (0, 1) or not self._in_bounds(x, y):
                    continue
                d = float(np.hypot(bx - x, by - y))
                if d < best_d:
                    best_d = d
                    nearest_team = int(team)
                    nearest_tid = int(tid)

            # strict possession: a player actually near the ball
            resolved: Optional[int] = nearest_team if best_d <= cfg.possession_radius_cm else None
            resolved_tid: Optional[int] = nearest_tid if best_d <= cfg.possession_radius_cm else None

            # effective owner — ĐÚNG luật thanh video (không carry-over):
            # resolved (≤3m) hoặc đội gần nhất (loose ≤60m), else None
            effective: Optional[int] = None
            if resolved is not None:
                effective = resolved
            elif nearest_team is not None and best_d <= cfg.possession_radius_cm * 20.0:
                effective = nearest_team

            # time weight of this frame (real seconds, capped against gaps)
            dt_f = 0.0 if prev_ok_t is None else min(fr.t - prev_ok_t, 0.6)
            prev_ok_t = fr.t
            sec = int(math.floor(fr.t))

            if effective is not None:
                owner_counts[effective] += 1
                owner_frames += 1
                timeline.setdefault(sec, [0.0, 0.0, 0.0])
                timeline[sec][effective] += dt_f
                # ball zone by length third (absolute x of the ball)
                z = 0 if bx < third else (1 if bx < 2.0 * third else 2)
                if resolved is not None:
                    zone_s[effective][z] += dt_f
            else:
                neutral_s += dt_f
                timeline.setdefault(sec, [0.0, 0.0, 0.0])
                timeline[sec][2] += dt_f

            # ---- player-level control & touches ------------------------- #
            if resolved_tid is not None:
                st = control.setdefault(resolved_tid, {"control_s": 0.0, "touches": 0})
                st["control_s"] += dt_f
                is_new = prev_res_tid != resolved_tid or (
                    prev_res_t is None or fr.t - prev_res_t > 0.4)
                if is_new:
                    st["touches"] += 1
                    prev_res_tid = resolved_tid
                prev_res_t = fr.t

            # ---- run continuity ----------------------------------------- #
            if effective is not None and effective == run_owner:
                pass  # run continues
            elif effective is not None:
                if run_start_t is not None and run_owner is not None:
                    runs.append((run_owner, run_start_t, prev_t if prev_t is not None else fr.t))
                run_owner = effective
                run_start_t = fr.t
            else:
                if run_start_t is not None and run_owner is not None and prev_t is not None:
                    if fr.t - prev_t > cfg.loose_break_s:
                        runs.append((run_owner, run_start_t, prev_t))
                        run_owner = None
                        run_start_t = None
            prev_t = fr.t

        if run_start_t is not None and run_owner is not None and prev_t is not None:
            runs.append((run_owner, run_start_t, prev_t))

        share = {}
        denom = max(1, owner_counts[0] + owner_counts[1])
        share[0] = round(100.0 * owner_counts[0] / denom, 1)
        share[1] = round(100.0 * owner_counts[1] / denom, 1)

        longest = {0: 0.0, 1: 0.0}
        run_durations: Dict[int, List[float]] = {0: [], 1: []}
        for team, s, e in runs:
            dur = max(0.0, e - s)
            longest[team] = max(longest[team], dur)
            run_durations[team].append(round(dur, 2))

        timeline_out = [
            {"t": s, "a": round(v[0], 2), "b": round(v[1], 2), "n": round(v[2], 2)}
            for s, v in sorted(timeline.items())
        ]

        return {
            "stat_frames": stat_frames,
            "owner_frames": owner_frames,
            "team_frame_counts": {0: owner_counts[0], 1: owner_counts[1]},
            "possession_share_pct": share,
            "longest_control_run_s": {str(k): round(v, 1) for k, v in longest.items()},
            "runs_count": {str(k): len(v) for k, v in run_durations.items()},
            "runs": {str(k): sorted(v, reverse=True)[:20] for k, v in run_durations.items()},
            "neutral_loose_est_s": round(neutral_s, 1),
            "timeline_s": timeline_out,
            "zone_s_3rds": {str(k): [round(x, 1) for x in v] for k, v in zone_s.items()},
            "player_control": {
                str(tid): {"control_s": round(v["control_s"], 1), "touches": v["touches"]}
                for tid, v in control.items()
            },
        }

    # ------------------------------------------------------------------ #
    # heat-map
    # ------------------------------------------------------------------ #

    def _analyze_heatmaps(self, team_of: Dict[int, int]) -> dict:
        cell = self.cfg.heatmap_cell_cm
        n_rows = int(math.ceil(self.length / cell))   # along length (x)
        n_cols = int(math.ceil(self.width / cell))    # along width (y)
        acc = {0: np.zeros((n_rows, n_cols)), 1: np.zeros((n_rows, n_cols))}

        for fr in self.frames:
            if not fr.ok:
                continue
            for tid, team, x, y in fr.players:
                if team not in (0, 1) or not self._in_bounds(x, y):
                    continue
                cx, cy = self._clamp(x, y)
                r = min(n_rows - 1, int(cx // cell))
                c = min(n_cols - 1, int(cy // cell))
                acc[team][r, c] += 1.0

        def to_lists(a: np.ndarray):
            return [[round(float(v), 1) for v in row] for row in a]

        # normalize each heat-map to [0..1] for plotting
        norm = {}
        for team in (0, 1):
            a = acc[team]
            mx = a.max() if a.max() > 0 else 1.0
            norm[team] = to_lists(a / mx)

        return {
            "cell_cm": cell,
            "rows_along_length": n_rows,
            "cols_along_width": n_cols,
            "team0_normalized": norm[0],
            "team1_normalized": norm[1],
        }

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #

    def analyze(self) -> dict:
        # ---- de-noise: per-frame common (homography) shift -------------- #
        # A wobbly homography shifts every projected coordinate by roughly the
        # same vector each frame. Estimate that common shift per frame (robust
        # median over players seen in both frames) and subtract its cumulative
        # effect from positions used for speed/distance. Relative distances
        # (possession) and heat-maps keep the raw absolute coordinates.
        n_f = len(self.frames)
        times = np.array([f.t for f in self.frames])
        shift = np.zeros((n_f, 2))
        prev: Dict[int, Tuple[float, float]] = {}
        for i, fr in enumerate(self.frames):
            cur = {tid: (x, y) for tid, team, x, y in fr.players if team in (0, 1)}
            if i > 0 and prev:
                common = list(cur.keys() & prev.keys())
                if len(common) >= 3:
                    ds = np.array(
                        [(cur[t][0] - prev[t][0], cur[t][1] - prev[t][1]) for t in common]
                    )
                    shift[i] = np.median(ds, axis=0)
            prev = cur
        cum_shift = np.cumsum(shift, axis=0)  # common displacement at each frame

        def correction_at(t: float) -> Tuple[float, float]:
            idx = int(np.searchsorted(times, t, side="right")) - 1
            idx = max(0, min(idx, n_f - 1))
            return float(cum_shift[idx, 0]), float(cum_shift[idx, 1])

        # ---- flicker-pair detection -------------------------------------- #
        # ByteTrack sometimes alternates two ids for the SAME player.  Two ids
        # are considered one person when they never appear together, yet take
        # turns at almost the same position within a short time.
        pres: Dict[int, List[float]] = {}
        pos_map: Dict[int, Dict[float, Tuple[float, float]]] = {}
        for fr in self.frames:
            for tid, team, x, y in fr.players:
                if team not in (0, 1) or not self._in_bounds(x, y):
                    continue
                tid = int(tid)
                pres.setdefault(tid, []).append(fr.t)
                pos_map.setdefault(tid, {})[fr.t] = (float(x), float(y))

        flicker_of: Dict[int, int] = {}
        ids = sorted(pres)
        for i in range(len(ids)):
            a = ids[i]
            if a in flicker_of:
                continue
            ta = np.array(pres[a])
            for j in range(i + 1, len(ids)):
                b = ids[j]
                if b in flicker_of:
                    continue
                tb = np.array(pres[b])
                # overlap?
                if not (ta.min() <= tb.max() and tb.min() <= ta.max()):
                    continue
                # never together?
                both = set(ta.round(4)) & set(tb.round(4))
                if len(both) > 0:
                    continue
                # median nearest-position distance while alternating
                ds = []
                pb = pos_map.get(b, {})
                for t_a in ta:
                    idx = int(np.searchsorted(tb, t_a))
                    cand = []
                    for k in (idx - 1, idx, idx + 1):
                        if 0 <= k < len(tb) and abs(tb[k] - t_a) <= self.cfg.flicker_time_window_s:
                            cand.append(pb.get(tb[k]))
                    if cand:
                        pa = pos_map.get(a, {}).get(t_a)
                        if pa is not None:
                            dmin = min(math.hypot(pa[0] - c[0], pa[1] - c[1]) for c in cand)
                            ds.append(dmin)
                if len(ds) >= self.cfg.flicker_min_samples and \
                        float(np.median(ds)) <= self.cfg.flicker_merge_dist_m * 100.0:
                    flicker_of[b] = a

        # ---- per-track observation lists + short-id re-identification ---- #
        # ByteTrack occasionally loses a player and later assigns a new id to
        # the same person.  If a brand-new id of a team reappears close (time +
        # distance) to a recently-lost id of that same team, we treat it as the
        # same player so touches/distance stay continuous.
        obs_by_track: Dict[int, List[Tuple[float, float, float]]] = {}
        raw_mean: Dict[int, List[float]] = {}       # id -> [sum_x, sum_y, count]
        team_of: Dict[int, int] = {}
        canon_of: Dict[int, int] = {}               # real id -> canonical id
        last_seen: Dict[int, Tuple[float, float, float, int]] = {}  # id -> (t,x,y,team)
        merge_gap = self.cfg.merge_max_gap_s
        merge_dist = self.cfg.merge_max_dist_m * 100.0

        for fr in self.frames:
            cx0, cy0 = correction_at(fr.t)
            seen_now = set()
            for tid, team, x, y in fr.players:
                if team not in (0, 1) or not self._in_bounds(x, y):
                    continue
                tid = int(flicker_of.get(int(tid), int(tid)))
                cur = canon_of.get(tid, tid)
                if tid not in canon_of:
                    # try to adopt a recently-lost canonical id of the same team
                    best_id = None
                    best_score = None
                    for cid, (lt, lx, ly, cteam) in last_seen.items():
                        if cteam != team or cid in seen_now:
                            continue
                        gap = fr.t - lt
                        if not (0.05 <= gap <= merge_gap):
                            continue
                        d = math.hypot(float(x) - lx, float(y) - ly)
                        if d <= merge_dist:
                            score = (gap, d)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_id = cid
                    if best_id is not None:
                        cur = best_id
                        canon_of[tid] = best_id
                seen_now.add(cur)
                obs_by_track.setdefault(cur, []).append(
                    (fr.t, float(x) - cx0, float(y) - cy0)
                )
                m = raw_mean.setdefault(cur, [0.0, 0.0, 0.0])
                m[0] += float(x)
                m[1] += float(y)
                m[2] += 1.0
                team_of.setdefault(cur, int(team))
                last_seen[cur] = (fr.t, float(x), float(y), int(team))

        rows: List[PlayerStatsRow] = []
        for tid, obs in obs_by_track.items():
            r = self._analyze_track(tid, obs)
            r.team = team_of.get(tid, -1)
            m = raw_mean.get(tid)
            if m and m[2] > 0:
                cx, cy = self._clamp(m[0] / m[2], m[1] / m[2])
                r.avg_x_m = round(cx / 100.0, 2)
                r.avg_y_m = round(cy / 100.0, 2)
            rows.append(r)

        possession = self._analyze_possession(team_of)

        # ---- normalize thirds by attack direction ------------------------ #
        # Ball-zone thirds are absolute along the pitch length, but whether a
        # third is the team's own goal area or the opponent's depends on which
        # side each team defends.  Own goal side is estimated from the tail of
        # each team's own player positions (defenders cluster near their goal).
        xs_by_team = {0: [], 1: []}
        for fr in self.frames:
            for _tid, team, x, y in fr.players:
                if team in (0, 1) and self._in_bounds(x, y):
                    xs_by_team[team].append(float(x))
        za: Dict[str, dict] = {}
        for team in (0, 1):
            xs = xs_by_team[team]
            own_low = True
            if len(xs) >= 5:
                p_low = float(np.percentile(xs, 3))
                p_high_mirror = self.length - float(np.percentile(xs, 97))
                own_low = p_low < p_high_mirror
            z = possession.get("zone_s_3rds", {}).get(str(team), [0, 0, 0])
            if own_low:
                za[str(team)] = {"own": z[0], "mid": z[1], "attack": z[2]}
            else:
                za[str(team)] = {"own": z[2], "mid": z[1], "attack": z[0]}
        possession["zone_attack_3rds"] = za
        heatmaps = self._analyze_heatmaps(team_of)

        # ---- team aggregations ------------------------------------------- #
        teams = {}
        for team in (0, 1):
            members = [r for r in rows if r.team == team]
            reliable = [r for r in members if r.speed_reliable]
            teams[str(team)] = {
                "players": len(members),
                "players_speed_reliable": len(reliable),
                "total_distance_m": round(sum(r.distance_m for r in members), 1),
                "avg_distance_m": round(
                    sum(r.distance_m for r in members) / len(members), 1) if members else 0.0,
                "avg_top_speed_kmh": round(
                    sum(r.top_speed_kmh for r in reliable) / len(reliable), 1) if reliable else 0.0,
                "total_sprints": sum(r.sprint_count for r in reliable),
                "total_active_s": round(sum(r.seconds_active for r in members), 1),
            }

        player_control = possession.get("player_control", {})
        control_by_canon: Dict[int, dict] = {}
        for rid, v in player_control.items():
            c = canon_of.get(int(rid), int(rid))
            acc = control_by_canon.setdefault(c, {"control_s": 0.0, "touches": 0})
            acc["control_s"] += float(v.get("control_s", 0.0))
            acc["touches"] += int(v.get("touches", 0))

        player_rows = []
        for r in rows:
            pc = control_by_canon.get(r.tracker_id, {})
            player_rows.append({
                "tracker_id": r.tracker_id,
                "team": r.team,
                "seconds_active": round(r.seconds_active, 2),
                "observations": r.n_obs,
                "distance_m": r.distance_m,
                "avg_speed_kmh": r.avg_speed_kmh,
                "top_speed_kmh": r.top_speed_kmh,
                "sprint_count": r.sprint_count,
                "sprint_distance_m": r.sprint_distance_m,
                "avg_x_m": r.avg_x_m,
                "avg_y_m": r.avg_y_m,
                "control_s": round(float(pc.get("control_s", 0.0)), 1),
                "touches": int(pc.get("touches", 0)),
                "speed_reliable": bool(getattr(r, "speed_reliable", True)),
            })
        player_rows.sort(key=lambda p: (-p["distance_m"], p["tracker_id"]))

        # sanity numbers
        duration = 0.0
        if self.frames:
            duration = max(0.0, self.frames[-1].t - self.frames[0].t)

        wobble_ms, quality = self._estimate_coordinate_quality()

        return {
            "video": dict(self.video_meta),
            "pitch_cm": {"length": self.length, "width": self.width},
            "processed": {
                "frames": len(self.frames),
                "duration_s": round(duration, 2),
                "ok_frames": sum(1 for f in self.frames if f.ok),
                "coordinate_wobble_est_ms": wobble_ms,
                "coordinate_quality": quality,
            },
            "config": {
                "possession_radius_cm": self.cfg.possession_radius_cm,
                "sprint_kmh": self.cfg.sprint_kmh,
                "sprint_min_duration_s": self.cfg.sprint_min_duration_s,
                "max_speed_kmh": self.cfg.max_speed_kmh,
            },
            "possession": possession,
            "teams": teams,
            "players": player_rows,
            "heatmap": heatmaps,
        }


# --------------------------------------------------------------------------- #
# Exporters
# --------------------------------------------------------------------------- #


class MatchStatsExporter:
    """Write the analysis summary to JSON / CSV / interactive HTML."""

    @staticmethod
    def export(
        stats: dict,
        out_dir: str | Path = "reports",
        basename: str = "match_stats",
    ) -> Dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{basename}.json"
        json_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        players_path = out_dir / f"{basename}_players.csv"
        MatchStatsExporter._write_players_csv(stats, players_path)

        team_path = out_dir / f"{basename}_teams.csv"
        MatchStatsExporter._write_teams_csv(stats, team_path)

        html_path = out_dir / f"{basename}.html"
        MatchStatsExporter._write_html(stats, html_path)

        files = {
            "json": json_path,
            "players_csv": players_path,
            "teams_csv": team_path,
            "html": html_path,
        }

        # ---- human-friendly Excel workbook (falls back to pretty HTML) --- #
        try:
            xlsx_path = out_dir / f"{basename}.xlsx"
            if MatchStatsExporter._write_xlsx(stats, xlsx_path):
                files["xlsx"] = xlsx_path
        except Exception:
            pass

        return files

    @staticmethod
    def _write_xlsx(stats: dict, path: Path) -> bool:
        """Write a styled .xlsx (sheets: summary, players, teams)."""
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment
            from openpyxl.utils import get_column_letter
        except Exception:
            return False

        wb = Workbook()
        head_fill = PatternFill("solid", fgColor="1F2937")
        head_font = Font(color="FFFFFF", bold=True)
        team_fills = {0: PatternFill("solid", fgColor="D6EEFF"),
                      1: PatternFill("solid", fgColor="FFD6EC")}

        def style_sheet(ws, headers, rows):
            ws.append(headers)
            for c in range(1, len(headers) + 1):
                cell = ws.cell(row=1, column=c)
                cell.fill = head_fill
                cell.font = head_font
                cell.alignment = Alignment(horizontal="center")
            for r in rows:
                ws.append(r)
            for i, h in enumerate(headers, start=1):
                width = max(len(str(h)), *(len(str(r[i - 1] if i - 1 < len(r) else ""))
                                          for r in rows)) if rows else len(str(h))
                ws.column_dimensions[get_column_letter(i)].width = min(width + 2, 40)
            ws.freeze_panes = "A2"

        # ---- sheet 1: summary -------------------------------------------- #
        ws = wb.active
        ws.title = "Tong quan"
        poss = stats["possession"]
        teams = stats["teams"]
        p = stats["processed"]
        rows = [
            ["Video", stats.get("video", {}).get("file", "?")],
            ["Khung hinh xu ly", p["frames"], "Thoi luong (s)", p["duration_s"]],
            ["Chat luong toa do", p.get("coordinate_quality", "?"),
             "Wobble (m/s)", p.get("coordinate_wobble_est_ms", "?")],
            [],
            ["Chi so", "Doi 0", "Doi 1"],
            ["Cam bong (%)", poss["possession_share_pct"].get(0),
             poss["possession_share_pct"].get(1)],
            ["Chuoi cam bong dai nhat (s)",
             poss["longest_control_run_s"].get("0"),
             poss["longest_control_run_s"].get("1")],
            ["Tong quang duong (m)",
             teams["0"].get("total_distance_m"), teams["1"].get("total_distance_m")],
            ["Tong buot toc (lan)",
             teams["0"].get("total_sprints"), teams["1"].get("total_sprints")],
            ["Toc do max TB (km/h)",
             teams["0"].get("avg_top_speed_kmh"), teams["1"].get("avg_top_speed_kmh")],
        ]
        for r in rows:
            ws.append(r)
        for i in range(1, 5):
            ws.column_dimensions[get_column_letter(i)].width = 30
        ws["A1"].font = Font(bold=True, size=13)

        # ---- sheet 2: players --------------------------------------------- #
        ws2 = wb.create_sheet("Cau thu")
        pcols = [
            "tracker_id", "team", "seconds_active", "observations",
            "distance_m", "avg_speed_kmh", "top_speed_kmh",
            "sprint_count", "sprint_distance_m", "control_s", "touches",
            "avg_x_m", "avg_y_m",
        ]
        phead = {
            "tracker_id": "#ID", "team": "Doi", "seconds_active": "Hoat dong (s)",
            "observations": "So frame", "distance_m": "Quang duong (m)",
            "avg_speed_kmh": "Toc do TB", "top_speed_kmh": "Toc do max",
            "sprint_count": "Buot toc", "sprint_distance_m": "QD buot toc (m)",
            "control_s": "Giu bong (s)", "touches": "Cham bong",
            "avg_x_m": "VT TB x", "avg_y_m": "VT TB y",
        }
        prows = []
        for pl in sorted(stats["players"], key=lambda x: x["tracker_id"]):
            prows.append([pl.get(c) for c in pcols])
        style_sheet(ws2, [phead[c] for c in pcols], prows)
        for r_i, pl in enumerate(sorted(stats["players"], key=lambda x: x["tracker_id"]), start=2):
            team = pl.get("team")
            if team in team_fills:
                ws2.cell(row=r_i, column=2).fill = team_fills[team]

        # ---- sheet 3: teams ------------------------------------------------ #
        ws3 = wb.create_sheet("Doi")
        zone = poss.get("zone_attack_3rds", {})
        tcols = [
            ("team", "Doi"), ("possession_share_pct", "Cam bong (%)"),
            ("players_tracked", "So cau thu"), ("total_distance_m", "Tong QD (m)"),
            ("avg_distance_m", "QD TB/cau thu"), ("avg_top_speed_kmh", "Top speed TB"),
            ("total_sprints", "Buot toc"), ("longest_control_run_s", "Chuoi dai nhat (s)"),
            ("own_third_s", "San nha (s)"), ("mid_third_s", "Giua san (s)"),
            ("attack_third_s", "San doi phuong (s)"),
        ]
        trows = []
        for team in ("0", "1"):
            t = teams.get(team, {})
            z = zone.get(team, {})
            trows.append([
                int(team), poss["possession_share_pct"].get(int(team)),
                t.get("players"), t.get("total_distance_m"), t.get("avg_distance_m"),
                t.get("avg_top_speed_kmh"), t.get("total_sprints"),
                poss["longest_control_run_s"].get(team),
                z.get("own", 0), z.get("mid", 0), z.get("attack", 0),
            ])
        style_sheet(ws3, [h for _, h in tcols], trows)
        for r_i, team in enumerate(("0", "1"), start=2):
            if int(team) in team_fills:
                ws3.cell(row=r_i, column=1).fill = team_fills[int(team)]

        wb.save(path)
        return True

    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_players_csv(stats: dict, path: Path) -> None:
        cols = [
            "tracker_id", "team", "seconds_active", "observations",
            "distance_m", "avg_speed_kmh", "top_speed_kmh",
            "sprint_count", "sprint_distance_m", "avg_x_m", "avg_y_m",
            "control_s", "touches", "speed_reliable",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(stats["players"])

    @staticmethod
    def _write_teams_csv(stats: dict, path: Path) -> None:
        rows = []
        poss = stats["possession"]
        zone = poss.get("zone_s_3rds", {"0": [0, 0, 0], "1": [0, 0, 0]})
        for team in ("0", "1"):
            t = stats["teams"].get(team, {})
            z = zone.get(team, [0, 0, 0])
            rows.append({
                "team": team,
                "possession_share_pct": poss["possession_share_pct"].get(int(team)),
                "players_tracked": t.get("players"),
                "total_distance_m": t.get("total_distance_m"),
                "avg_distance_m": t.get("avg_distance_m"),
                "avg_top_speed_kmh": t.get("avg_top_speed_kmh"),
                "total_sprints": t.get("total_sprints"),
                "longest_control_run_s": poss["longest_control_run_s"].get(str(team)),
                "ball_control_1st_third_s": z[0],
                "ball_control_middle_third_s": z[1],
                "ball_control_last_third_s": z[2],
            })
        cols = list(rows[0].keys()) if rows else ["team"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------ #
    # HTML
    # ------------------------------------------------------------------ #

    @staticmethod
    def _plotly_js() -> str:
        """Return inline plotly.min.js if available locally, else a CDN tag."""
        try:
            import plotly
            p = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
            if p.exists():
                return f"<script>{p.read_text(encoding='utf-8')}</script>"
        except Exception:
            pass
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    @staticmethod
    def _write_html(stats: dict, path: Path) -> None:
        poss = stats["possession"]
        teams = stats["teams"]
        players = stats["players"]
        heat = stats["heatmap"]

        video_name = stats.get("video", {}).get("file", "?")
        frames = stats["processed"]["frames"]
        dur = stats["processed"]["duration_s"]

        share0 = poss["possession_share_pct"].get(0, 0)
        share1 = poss["possession_share_pct"].get(1, 0)
        t0 = teams.get("0", {})
        t1 = teams.get("1", {})

        top_sprinters = sorted(
            [p for p in players if p["sprint_count"] > 0],
            key=lambda p: p["sprint_count"], reverse=True)[:10]
        fastest = sorted(players, key=lambda p: p["top_speed_kmh"], reverse=True)[:10]
        runners = players[:10]

        def team_color(team: int) -> str:
            return "#00BFFF" if team == 0 else "#FF1493"

        js_players = json.dumps(players, ensure_ascii=False)
        js_heat0 = json.dumps(heat["team0_normalized"])
        js_heat1 = json.dumps(heat["team1_normalized"])

        # pitch outline (cm) for overlaying the heat-map
        length = stats["pitch_cm"]["length"]
        width = stats["pitch_cm"]["width"]
        pitch = SoccerPitchConfiguration()
        verts = pitch.vertices

        html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Match Statistics Report</title>
{MatchStatsExporter._plotly_js()}
<style>
  body {{ font-family: 'Segoe UI', Roboto, Arial, sans-serif; margin: 0; background: #101418; color: #e8edf2; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 18px; }}
  h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin: 26px 0 6px; }}
  .muted {{ color: #8b98a5; font-size: 12px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }}
  .card {{ background: #1a2129; border: 1px solid #2a3540; border-radius: 10px; padding: 12px 16px; min-width: 150px; }}
  .card .k {{ color: #8b98a5; font-size: 12px; }} .card .v {{ font-size: 22px; font-weight: 600; }}
  .pill {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #2a3540; padding: 5px 8px; text-align: left; }}
  th {{ color:#8b98a5; font-weight: 500; }}
  .note {{ background:#17212b; border-left: 3px solid #3b82f6; padding: 8px 12px; font-size: 12px; margin: 10px 0; color:#aab6c2; }}
</style></head><body><div class="wrap">
<h1>⚽ Báo cáo chỉ số trận đấu</h1>
<div class="muted">Video: {video_name} &nbsp;|&nbsp; {frames} khung hình xử lý &nbsp;|&nbsp; {dur:.1f}s hiệu lực</div>

<div class="cards">
  <div class="card"><div class="k"><span class="pill" style="background:#00BFFF"></span>Đội 0 cầm bóng</div><div class="v">{share0:.1f}%</div></div>
  <div class="card"><div class="k"><span class="pill" style="background:#FF1493"></span>Đội 1 cầm bóng</div><div class="v">{share1:.1f}%</div></div>
  <div class="card"><div class="k">Đội 0 · bứt tốc / quãng đường</div><div class="v">{t0.get('total_sprints',0)} lần · {t0.get('total_distance_m',0)} m</div></div>
  <div class="card"><div class="k">Đội 1 · bứt tốc / quãng đường</div><div class="v">{t1.get('total_sprints',0)} lần · {t1.get('total_distance_m',0)} m</div></div>
</div>
<div class="note">⚠️ Chỉ số được <b>ước lượng từ thị giác máy tính</b>: cầm bóng = đội có cầu thủ gần bóng nhất ≤ {stats['config']['possession_radius_cm']:.0f}cm;
tốc độ suy từ toạ độ chiếu homography (cửa sổ ~{stats['config'].get('speed_window_s', 0.5):.1f}s);
bứt tốc = tốc độ ≥ {stats['config']['sprint_kmh']:.0f} km/h duy trì ≥ {stats['config']['sprint_min_duration_s']:.1f}s.
Chất lượng toạ độ video này: <b>{stats['processed'].get('coordinate_quality', '?')}</b>
(wobble ~{stats['processed'].get('coordinate_wobble_est_ms', '?')} m/s) — wobble càng cao, sai số tốc độ/quãng đường càng lớn.</div>

<div id="pos"></div>
<h2>⏱️ Biến động cầm bóng theo giây (luỹ kế)</h2><div id="timeline"></div>
<h2>📈 Cầu thủ bứt tốc nhiều nhất</h2><div id="sprinters"></div>
<h2>🚀 Cầu thủ nhanh nhất</h2><div id="fastest"></div>
<h2>🏃 Quãng đường di chuyển nhiều nhất</h2><div id="runners"></div>
<h2>🗺️ Vị trí kiểm soát bóng theo 1/3 sân (dọc)</h2><div id="zone"></div>
<h2>🗺️ Bản đồ nhiệt hoạt động (Đội 0 xanh / Đội 1 hồng)</h2><div id="heat"></div>
<h2>📋 Chi tiết từng cầu thủ</h2><div id="table"></div>

<script>
const P = {js_players};
const TEAM = c => c === 0 ? '#00BFFF' : '#FF1493';
function barChart(el, title, rows, key, label) {{
  const r = rows.slice().sort((a,b)=>b[key]-a[key]).slice(0,10);
  Plotly.newPlot(el, [{{
    type:'bar', orientation:'h',
    x: r.map(o=>o[key]), y: r.map(o=>'#'+o.tracker_id),
    marker: {{ color: r.map(o=>TEAM(o.team)) }},
    text: r.map(o=>o[key].toFixed(1)+' '+label), textposition:'outside'
  }}], {{ title, height: 320, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{{color:'#c8d2db'}}, margin:{{l:50,r:120,t:40,b:30}},
    xaxis:{{gridcolor:'#2a3540'}}, yaxis:{{autorange:'reversed'}} }}, {{displayModeBar:false}});
}}
// possession donut
Plotly.newPlot('pos', [{{
  type:'pie', labels:['Đội 0','Đội 1'],
  values:[{share0},{share1}], hole:0.45,
  marker:{{colors:['#00BFFF','#FF1493']}},
  textinfo:'label+percent'
}}], {{ title:'Tỉ lệ cầm bóng', height: 320, font:{{color:'#c8d2db'}}, paper_bgcolor:'rgba(0,0,0,0)' }}, {{displayModeBar:false}});

// ---- possession timeline (cumulative %) ---------------------------------- //
const TL = {json.dumps(poss.get('timeline_s', []))};
let ca = 0, cb = 0;
const tx=[], tyA=[], tyB=[];
TL.forEach(b => {{
  ca += b.a; cb += b.b;
  tx.push(b.t);
  const tot = Math.max(ca + cb, 1e-6);
  tyA.push(100 * ca / tot); tyB.push(100 * cb / tot);
}});
Plotly.newPlot('timeline', [
  {{ x: tx, y: tyA, type:'scatter', mode:'lines+markers', name:'Đội 0', line:{{color:'#00BFFF'}} }},
  {{ x: tx, y: tyB, type:'scatter', mode:'lines+markers', name:'Đội 1', line:{{color:'#FF1493'}} }}
], {{ title:'Phần trăm cầm bóng luỹ kế theo thời gian', height: 320,
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{{color:'#c8d2db'}},
  xaxis:{{title:'giây', gridcolor:'#2a3540'}}, yaxis:{{title:'%', range:[0,100], gridcolor:'#2a3540'}} }}, {{displayModeBar:false}});

// ---- ball control zones, normalized by attack direction ------------------- //
// zone_attack_3rds: each team's ball time in [own third, middle, opponent third]
const ZA = {json.dumps(poss.get('zone_attack_3rds', {'0': {'own':0,'mid':0,'attack':0}, '1': {'own':0,'mid':0,'attack':0}}))};
const zx = ['Sân nhà (1/3 phòng ngự)','Giữa sân','Sân đối phương (1/3 tấn công)'];
const zA = [ZA['0'].own, ZA['0'].mid, ZA['0'].attack];
const zB = [ZA['1'].own, ZA['1'].mid, ZA['1'].attack];
Plotly.newPlot('zone', [
  {{ type:'bar', name:'Đội 0', x: zx, y: zA, marker:{{color:'#00BFFF'}} }},
  {{ type:'bar', name:'Đội 1', x: zx, y: zB, marker:{{color:'#FF1493'}} }}
], {{ barmode:'group', title:'Thời gian kiểm soát bóng theo vị trí so với hướng tấn công (giây)',
  height: 320, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{{color:'#c8d2db'}}, yaxis:{{gridcolor:'#2a3540'}} }}, {{displayModeBar:false}});

barChart('sprinters', 'Số lần bứt tốc', P, 'sprint_count', 'lần');
barChart('fastest',   'Tốc độ tối đa (km/h)', P, 'top_speed_kmh', 'km/h');
barChart('runners',   'Quãng đường (m)', P, 'distance_m', 'm');

// heat-maps with pitch outline (axes in cm: x = length, y = width)
const PITCH_V = {json.dumps(verts)};
const PITCH_E = {json.dumps(pitch.edges)};
const CELL = {heat['cell_cm']};
const L = {length}, W = {width};
function outlineTraces() {{
  const tr = [];
  PITCH_E.forEach(([a,b]) => {{
    const p1 = PITCH_V[a-1], p2 = PITCH_V[b-1];
    tr.push({{type:'scatter', mode:'lines', x:[p1[0],p2[0]], y:[p1[1],p2[1]],
      line:{{color:'#ffffff', width:1.2}}, hoverinfo:'skip', showlegend:false}});
  }});
  const R=915, cx=L/2, cy=W/2, pts=[];
  for (let i=0;i<=120;i++) {{ const a=2*Math.PI*i/120; pts.push([cx+R*Math.cos(a), cy+R*Math.sin(a)]); }}
  tr.push({{type:'scatter', mode:'lines', x:pts.map(p=>p[0]), y:pts.map(p=>p[1]),
    line:{{color:'#ffffff', width:1.2}}, hoverinfo:'skip', showlegend:false}});
  return tr;
}}
function heatTrace(zraw, color) {{
  // zraw: row index = length bin (x), col index = width bin (y)
  const nR = zraw.length, nC = zraw[0].length;
  const x = Array.from({{length:nR}}, (_,r) => r*CELL + CELL/2);   // length cm
  const y = Array.from({{length:nC}}, (_,c) => c*CELL + CELL/2);   // width cm
  const zT = zraw[0].map((_,c) => zraw.map(row => row[c]));        // transpose -> rows = width
  return {{ type:'heatmap', x:x, y:y, z:zT,
    colorscale:[[0,'rgba(0,0,0,0)'],[1,color]], showscale:false,
    zsmooth:'best', hoverinfo:'skip', opacity:0.85 }};
}}
const heatData = [
  heatTrace({json.dumps(heat['team0_normalized'])}, '#00BFFF'),
  heatTrace({json.dumps(heat['team1_normalized'])}, '#FF1493')
].concat(outlineTraces());
const heatLayout = {{
  title:'Bản đồ nhiệt hoạt động · trục x = chiều dài sân (m), y = chiều rộng sân (m)',
  height:480, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#1e8c33',
  xaxis:{{range:[0,L], showgrid:false, zeroline:false, visible:false}},
  yaxis:{{range:[0,W], showgrid:false, zeroline:false, visible:false, scaleanchor:'x', scaleratio:1}},
  margin:{{l:20,r:20,t:50,b:20}}
}};
Plotly.newPlot('heat', heatData, heatLayout, {{displayModeBar:false}});

// player table
const head = ['#ID','Đội','Giữ bóng (s)','Chạm bóng','Thời gian (s)','Quãng (m)','Tốc TB','Tốc max','Bứt tốc','VT TB x','VT TB y'];
const body = P.map(p => ['#'+p.tracker_id, 'Đội '+p.team,
  (p.control_s||0).toFixed(1), p.touches||0,
  p.seconds_active.toFixed(1), p.distance_m.toFixed(1),
  p.avg_speed_kmh.toFixed(1), p.top_speed_kmh.toFixed(1), p.sprint_count,
  p.avg_x_m.toFixed(1), p.avg_y_m.toFixed(1)]);
Plotly.newPlot('table', [{{ type:'table',
  header:{{values:head, fill:{{color:'#232e3a'}}, font:{{color:'#e8edf2', size:11}}, align:'left'}},
  cells:{{values: body[0].map((_,i)=>body.map(r=>r[i])), fill:{{color:'#182027'}},
    font:{{color:'#e8edf2', size:11}}, align:'left'}}
}}], {{ title:'Cầu thủ', height: 420, paper_bgcolor:'rgba(0,0,0,0)', font:{{color:'#c8d2db'}} }}, {{displayModeBar:false}});
</script>
</div></body></html>"""
        path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Convenience API
# --------------------------------------------------------------------------- #


def analyze_and_export(
    recorder: MatchStatsRecorder,
    pitch_config: SoccerPitchConfiguration,
    out_dir: str | Path = "reports",
    basename: str = "match_stats",
    video_meta: Optional[dict] = None,
    config: Optional[MatchStatsConfig] = None,
) -> Dict[str, Path]:
    """Run the analysis on recorded frames and write every report file."""
    analyzer = MatchStatsAnalyzer(
        frames=recorder.frames,
        pitch_config=pitch_config,
        config=config or recorder.config,
        video_meta=video_meta or {},
    )
    stats = analyzer.analyze()
    return MatchStatsExporter.export(stats, out_dir=out_dir, basename=basename)
