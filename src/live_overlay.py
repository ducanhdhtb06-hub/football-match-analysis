"""
Live on-video overlay: possession scoreboard + per-player speed labels.

While the pipeline processes a video this small stateful helper keeps a short
history of (time, de-noised pitch position) per tracked player and computes a
current speed over a ~0.5 s window - the same robust recipe as match_stats.py
(common-shift removal + window velocity).  It then annotates every frame with:

    * a live possession bar (rolling 30 s window) at the top,
    * the current speed (km/h) of every tracked outfield player/GK,
    * a highlighted label while a player is sprinting (>= sprint_kmh).

Thresholds are shared with MatchStatsConfig.  Text drawn with cv2 must be
pure ASCII (Hershey fonts do not support Unicode).
"""

from __future__ import annotations

from collections import deque
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .match_stats import MatchStatsConfig

# BGR colors (match VisualConfig)
_TEAM0_BGR = (255, 191, 0)      # deep sky blue  #00BFFF
_TEAM1_BGR = (147, 20, 255)     # deep pink      #FF1493
_SPRINT_BGR = (0, 80, 255)      # orange-red
_BG_BGR = (23, 26, 30)
_WHITE = (245, 245, 245)


class LiveStatsOverlay:
    """Stateful per-frame overlay used by the pipeline video loop."""

    # rolling possession window (seconds) shown on the bar
    POSSESSION_WINDOW_S = 1e9   # thanh hiển thị luỹ kế TOÀN BỘ video (khớp báo cáo)

    def __init__(self, config: Optional[MatchStatsConfig] = None):
        self.cfg = config or MatchStatsConfig()
        self._vel: Dict[int, list] = {}          # tid -> [t, x, y, vx_cms, vy_cms]
        self._prev_raw: Dict[int, Tuple[float, float]] = {}
        self._cum_c = np.zeros(2)                # cumulative common shift (cm)
        self._possession: deque = deque()        # (t_sec, owner: 0/1/None)
        self._t_now = 0.0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Call at the start of each new video."""
        self._vel.clear()
        self._prev_raw.clear()
        self._cum_c = np.zeros(2)
        self._possession.clear()
        self._t_now = 0.0

    # ------------------------------------------------------------------ #
    def update_frame(
        self,
        t_sec: float,
        players_raw: Dict[int, Tuple[float, float]],
        ok: bool,
    ) -> None:
        """Feed the raw projected positions of one processed frame.

        players_raw: tracker_id -> (x_cm, y_cm) in the wobbling homography
        frame; the common per-frame wobble is estimated and removed so the
        stored histories are stable.
        """
        self._t_now = t_sec

        if not ok:
            self._vel.clear()
            self._prev_raw = {}
            return

        # ---- estimate + integrate the common (homography) shift --------- #
        if self._prev_raw:
            common = list(set(players_raw) & set(self._prev_raw))
            if len(common) >= 3:
                ds = np.array(
                    [[players_raw[t][0] - self._prev_raw[t][0],
                      players_raw[t][1] - self._prev_raw[t][1]] for t in common]
                )
                self._cum_c += np.median(ds, axis=0)
        self._prev_raw = dict(players_raw)

        # ---- per-player velocity (EMA over corrected positions) -------- #
        gate_cm = self.cfg.jump_gate_m * 100.0
        for tid, (rx, ry) in players_raw.items():
            cx, cy = rx - float(self._cum_c[0]), ry - float(self._cum_c[1])
            prev = self._vel.get(tid)
            vx = vy = 0.0
            if prev is not None:
                dtc = t_sec - prev[0]
                if 0.0 < dtc < 2.0:
                    if math.hypot(cx - prev[1], cy - prev[2]) > gate_cm:
                        pass                       # glitch: restart from 0
                    else:
                        k = 0.35
                        vx = prev[3] + k * ((cx - prev[1]) / dtc - prev[3])
                        vy = prev[4] + k * ((cy - prev[2]) / dtc - prev[4])
            self._vel[tid] = [t_sec, cx, cy, vx, vy]

    def set_possession_owner(self, owner: Optional[int]) -> None:
        """Register the frame possession owner (0/1/None), resolved by caller."""
        self._possession.append((self._t_now, owner))
        while self._possession and \
                self._possession[0][0] < self._t_now - self.POSSESSION_WINDOW_S:
            self._possession.popleft()

    def possession_share(self) -> Tuple[float, float]:
        """(team0 %, team1 %) over the rolling window (50/50 when unknown)."""
        c0 = sum(1 for _, o in self._possession if o == 0)
        c1 = sum(1 for _, o in self._possession if o == 1)
        total = c0 + c1
        if total == 0:
            return 50.0, 50.0
        return 100.0 * c0 / total, 100.0 * c1 / total

    # ------------------------------------------------------------------ #
    def _current_speed_ms(self, tid: int) -> float:
        """Smoothed velocity magnitude (m/s) from the EMA state."""
        st = self._vel.get(tid)
        if st is None:
            return 0.0
        v_ms = math.hypot(st[3], st[4]) / 100.0
        cap = self.cfg.max_speed_kmh / 3.6
        return min(v_ms, cap)

    # ------------------------------------------------------------------ #
    def annotate(
        self,
        scene: np.ndarray,
        players_px: Dict[int, Tuple[int, int, int]],  # tid -> (cx_px, top_px, team)
    ) -> np.ndarray:
        """Draw the possession bar and per-player speed labels."""
        out = scene.copy()
        h_img, w_img = out.shape[:2]

        # ---- per-player speed labels ------------------------------------ #
        for tid, (cx, top, team) in players_px.items():
            v_ms = self._current_speed_ms(tid)
            if v_ms < 0.5:
                continue
            kmh = v_ms * 3.6
            sprint = kmh >= self.cfg.sprint_kmh
            text = f"{kmh:.0f}" if not sprint else f"*{kmh:.0f}"
            color = _SPRINT_BGR if sprint else (
                _TEAM0_BGR if team == 0 else _TEAM1_BGR)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            x0 = max(2, min(int(cx) - tw // 2, w_img - tw - 4))
            y0 = max(int(top) - 8 - th, 4)
            cv2.rectangle(out, (x0, y0 - 2), (x0 + tw + 2, y0 + th + 3), color, -1)
            cv2.putText(out, text, (x0 + 1, y0 + th), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (10, 10, 10), 1, cv2.LINE_AA)

        # ---- possession scoreboard (top centre) -------------------------- #
        s0, s1 = self.possession_share()
        label_a = f"BALL A {s0:.0f}%"
        label_b = f"B {s1:.0f}%"
        fs = 0.6 if w_img > 1200 else 0.48
        (aw, ah), _ = cv2.getTextSize(label_a, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        (bw, bh), _ = cv2.getTextSize(label_b, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        bar_w = min(aw + bw + 70, int(w_img * 0.62))
        x = max(6, (w_img - bar_w) // 2)
        y = 10
        box_h = max(ah, bh) + 22
        cv2.rectangle(out, (x - 4, y - 4), (x + bar_w + 4, y + box_h), _BG_BGR, -1)
        xa = x + int(bar_w * s0 / 100.0)
        cv2.rectangle(out, (x, y + 2), (xa, y + 9), _TEAM0_BGR, -1)
        cv2.rectangle(out, (xa, y + 2), (x + bar_w, y + 9), _TEAM1_BGR, -1)
        cv2.putText(out, label_a, (x, y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, _WHITE, 2, cv2.LINE_AA)
        cv2.putText(out, label_b, (x + bar_w - bw, y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, _WHITE, 2, cv2.LINE_AA)

        cv2.putText(out, f"t={self._t_now:.1f}s", (w_img - 170, h_img - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
        return out
