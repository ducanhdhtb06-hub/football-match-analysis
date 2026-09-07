import warnings
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*KeyPoints.confidence.*")

from .config import ModelConfig, ClassIDConfig, VisualConfig, PipelineConfig
from .pitch_config import SoccerPitchConfiguration
from .detection import PlayerDetector, PlayerTracker
from .team_classifier import TeamClassifier, resolve_goalkeepers_team_id, collect_player_crops
from .pitch_detector import PitchKeypointDetector
from .projection import PitchProjector, ViewTransformer
from .pitch_annotator import PitchDrawer
from .visualizer import VideoFrameVisualizer
from .match_stats import (
    MatchStatsConfig,
    MatchStatsRecorder,
    MatchStatsAnalyzer,
    MatchStatsExporter,
)
from .live_overlay import LiveStatsOverlay


def get_video_codec(filename: str) -> str:
    """
    Select appropriate OpenCV FourCC codec string based on video file extension.
    - .webm: uses 'VP80' (Google VP8)
    - .avi:  uses 'XVID'
    - .mp4 / .mov / others: uses 'mp4v'
    """
    ext = Path(filename).suffix.lower()
    if ext == ".webm":
        return "VP80"
    elif ext == ".avi":
        return "XVID"
    else:
        return "mp4v"


class FootballAnalysisPipeline:
    """
    End-to-end video analysis pipeline:
    1. Player/Ball/Referee detection with YOLO.
    2. Multi-object tracking with ByteTrack.
    3. Player team classification using SigLIP + KMeans.
    4. Goalkeeper team assignment based on spatial proximity.
    5. Pitch keypoint detection and homography projection.
    6. 2D Radar & Voronoi visualization and mini-map compositing.
    7. Dual support for .mp4 and .webm video formats.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        class_config: Optional[ClassIDConfig] = None,
        visual_config: Optional[VisualConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        pitch_config: Optional[SoccerPitchConfiguration] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.class_config = class_config or ClassIDConfig()
        self.visual_config = visual_config or VisualConfig()
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.pitch_config = pitch_config or SoccerPitchConfiguration()

        self.detector = PlayerDetector(
            model_config=self.model_config,
            class_config=self.class_config,
            visual_config=self.visual_config,
        )
        self.tracker = PlayerTracker()
        self.team_classifier = TeamClassifier(
            device=self.model_config.device,
            batch_size=self.pipeline_config.batch_size,
            model_path=self.model_config.siglip_model_path,
        )
        self.pitch_detector = PitchKeypointDetector(
            model_config=self.model_config,
            pitch_config=self.pitch_config,
        )
        self.pitch_projector = PitchProjector(
            cache_homography=self.pipeline_config.cache_homography,
            smooth_homography=self.pipeline_config.smooth_homography,
            homography_smoothing_alpha=self.pipeline_config.homography_smoothing_alpha,
            stickiness_cm=self.pipeline_config.homography_stickiness_cm,
            rebaseline_frames=self.pipeline_config.homography_rebaseline_frames,
        )
        self.pitch_drawer = PitchDrawer(
            config=self.pitch_config,
            visual_config=self.visual_config,
        )
        self.frame_visualizer = VideoFrameVisualizer(
            visual_config=self.visual_config
        )
        self._kp_counter = 0
        self._last_key_points = None

    def fit_teams_from_video(self, video_path: str) -> None:
        """
        Sample frames from video to collect player crops and train the team clustering model.
        """
        print(f"Collecting player crops from: {video_path} (stride={self.pipeline_config.crop_stride})...")
        crops = collect_player_crops(
            video_path=video_path,
            detector=self.detector,
            player_id=self.class_config.player_id,
            stride=self.pipeline_config.crop_stride,
            max_crops=self.pipeline_config.max_fitting_crops,
        )
        if len(crops) < 2:
            print("[Warning] Too few player crops collected to fit team classifier. Will use default team 0.")
            return

        print(f"Fitting team classifier on {len(crops)} crops...")
        self.team_classifier.fit(crops)
        print("Team classifier fitting complete!")

    def process_frame(
        self,
        frame: np.ndarray,
        enable_voronoi: Optional[bool] = None,
        stats_recorder: Optional[MatchStatsRecorder] = None,
        frame_time: float = 0.0,
        live_overlay: Optional[LiveStatsOverlay] = None,
    ) -> np.ndarray:
        """Process a single frame and return the annotated frame with radar mini-map."""
        use_voronoi = self.pipeline_config.enable_voronoi if enable_voronoi is None else enable_voronoi

        # 1. Detection
        ball_detections, non_ball_detections = self.detector.detect(frame)

        # 2. Tracking
        tracked_detections = self.tracker.update(non_ball_detections)

        # 3. Entity separation
        gk_mask = tracked_detections.class_id == self.class_config.goalkeeper_id
        pl_mask = tracked_detections.class_id == self.class_config.player_id
        ref_mask = tracked_detections.class_id == self.class_config.referee_id

        gk_detections = tracked_detections[gk_mask]
        player_detections = tracked_detections[pl_mask]
        referee_detections = tracked_detections[ref_mask]

        # 3b. Drop tiny (far-field) boxes that destabilize tracking / metrics
        min_h = self.pipeline_config.min_entity_height_px

        def _filter_small(dets: sv.Detections) -> sv.Detections:
            if len(dets) == 0:
                return dets
            keep = (dets.xyxy[:, 3] - dets.xyxy[:, 1]) >= min_h
            return dets[keep] if not keep.all() else dets

        if min_h > 0:
            player_detections = _filter_small(player_detections)
            gk_detections = _filter_small(gk_detections)
            referee_detections = _filter_small(referee_detections)

        # 4. Team classification for players
        if len(player_detections) > 0 and self.team_classifier.is_fitted:
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
            player_detections.class_id = self.team_classifier.predict(player_crops)
        elif len(player_detections) > 0:
            player_detections.class_id = np.zeros(len(player_detections), dtype=int)

        # 5. Resolve goalkeeper team ID
        if len(gk_detections) > 0:
            gk_detections.class_id = resolve_goalkeepers_team_id(
                players=player_detections,
                goalkeepers=gk_detections
            )

        # 6. Map referee class ID to index 2 (matching VisualConfig color palette)
        if len(referee_detections) > 0:
            referee_detections.class_id = np.full(len(referee_detections), 2, dtype=int)

        # 7. Merge tracked entities for main frame annotation
        merged_entities = []
        if len(player_detections) > 0:
            merged_entities.append(player_detections)
        if len(gk_detections) > 0:
            merged_entities.append(gk_detections)
        if len(referee_detections) > 0:
            merged_entities.append(referee_detections)

        if merged_entities:
            all_tracked = sv.Detections.merge(merged_entities)
        else:
            all_tracked = sv.Detections.empty()

        annotated_frame = self.frame_visualizer.annotate(
            frame=frame,
            all_detections=all_tracked,
            ball_detections=ball_detections,
        )

        # 8. Pitch keypoint detection & Homography projection
        # (run the remote keypoint model only every `pitch_keypoint_every`
        # frames; in between reuse the last result -> fewer API calls)
        if self._kp_counter % max(1, self.pipeline_config.pitch_keypoint_every) == 0:
            self._last_key_points = self.pitch_detector.detect(frame)
        self._kp_counter += 1
        key_points = self._last_key_points
        frame_ref, pitch_ref, _ = self.pitch_detector.get_reference_points(key_points)
        transformer = self.pitch_projector.update(frame_ref, pitch_ref)

        pitch_ball_xy = self.pitch_projector.project_detections(transformer, ball_detections)

        all_players_list = []
        if len(player_detections) > 0:
            all_players_list.append(player_detections)
        if len(gk_detections) > 0:
            all_players_list.append(gk_detections)

        all_players = None
        if all_players_list:
            all_players = sv.Detections.merge(all_players_list)
            team_0_players = all_players[all_players.class_id == 0]
            team_1_players = all_players[all_players.class_id == 1]
            pitch_team_0_xy = self.pitch_projector.project_detections(transformer, team_0_players)
            pitch_team_1_xy = self.pitch_projector.project_detections(transformer, team_1_players)
        else:
            pitch_team_0_xy = np.empty((0, 2), dtype=np.float32)
            pitch_team_1_xy = np.empty((0, 2), dtype=np.float32)

        pitch_referee_xy = self.pitch_projector.project_detections(transformer, referee_detections)

        # 9. Generate 2D pitch radar view & overlay onto main frame
        radar_view = self.pitch_drawer.build_radar_view(
            ball_xy=pitch_ball_xy,
            team_0_xy=pitch_team_0_xy,
            team_1_xy=pitch_team_1_xy,
            referee_xy=pitch_referee_xy,
            enable_voronoi=use_voronoi,
        )

        final_frame = self.pitch_drawer.overlay_mini_map(annotated_frame, radar_view)

        # 10. Optional: record this frame for post-match statistics
        if stats_recorder is not None:
            self._record_stats_frame(
                recorder=stats_recorder,
                t_sec=frame_time,
                transformer=transformer,
                all_players=all_players,
                ball_xy=pitch_ball_xy,
            )

        # 11. Optional: live overlay (possession bar + player speed labels)
        if live_overlay is not None:
            final_frame = self._apply_live_overlay(
                frame=final_frame,
                overlay=live_overlay,
                t_sec=frame_time,
                transformer=transformer,
                all_players=all_players,
                ball_xy=pitch_ball_xy,
            )

        return final_frame

    def _record_stats_frame(
        self,
        recorder: MatchStatsRecorder,
        t_sec: float,
        transformer: Optional[ViewTransformer],
        all_players: Optional[sv.Detections],
        ball_xy: np.ndarray,
    ) -> None:
        """Collect projected player/ball positions for the stats engine."""
        if transformer is None:
            recorder.record(t_sec=t_sec, ball_xy=None, players=[], ok=False)
            return

        players: List[tuple] = []
        if (
            all_players is not None
            and len(all_players) > 0
            and all_players.tracker_id is not None
        ):
            projected = self.pitch_projector.project_detections(transformer, all_players)
            for tid, team, pt in zip(all_players.tracker_id, all_players.class_id, projected):
                x, y = float(pt[0]), float(pt[1])
                if np.isfinite(x) and np.isfinite(y):
                    players.append((int(tid), int(team), x, y))

        ball = None
        if len(ball_xy) > 0:
            bx, by = float(ball_xy[0][0]), float(ball_xy[0][1])
            if np.isfinite(bx) and np.isfinite(by):
                ball = (bx, by)

        recorder.record(t_sec=t_sec, ball_xy=ball, players=players, ok=True)

    def _apply_live_overlay(
        self,
        frame: np.ndarray,
        overlay: LiveStatsOverlay,
        t_sec: float,
        transformer: Optional[ViewTransformer],
        all_players: Optional[sv.Detections],
        ball_xy: np.ndarray,
    ) -> np.ndarray:
        """Update overlay state with this frame and draw scoreboard + speeds."""
        raw_players: dict = {}
        teams: dict = {}
        px_players: dict = {}

        if transformer is not None and all_players is not None and len(all_players) > 0:
            tracker_id = all_players.tracker_id
            if tracker_id is not None:
                proj = self.pitch_projector.project_detections(transformer, all_players)
                bottom = all_players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                top_y = all_players.xyxy[:, 1]
                for i in range(len(all_players)):
                    t = int(tracker_id[i])
                    x, y = float(proj[i, 0]), float(proj[i, 1])
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    team = int(all_players.class_id[i])
                    raw_players[t] = (x, y)
                    teams[t] = team
                    px_players[t] = (float(bottom[i, 0]), float(top_y[i]), team)

        overlay.update_frame(t_sec=t_sec, players_raw=raw_players,
                             ok=transformer is not None)

        # ---- live possession owner (nearest team player to the ball) ---- #
        owner: Optional[int] = None
        if transformer is not None and len(ball_xy) > 0:
            bx, by = float(ball_xy[0][0]), float(ball_xy[0][1])
            if np.isfinite(bx) and np.isfinite(by):
                best_team = None
                best_d = float("inf")
                for t, (x, y) in raw_players.items():
                    d = float(np.hypot(bx - x, by - y))
                    if d < best_d:
                        best_d = d
                        best_team = teams.get(t)
                if best_team is not None and best_d <= overlay.cfg.possession_radius_cm * 20.0:
                    owner = int(best_team)   # cùng luật "đội gần nhất" như báo cáo cuối
        overlay.set_possession_owner(owner)

        return overlay.annotate(frame, px_players)

    def process_video(
        self,
        source_path: str,
        target_path: str,
        max_frames: Optional[int] = None,
        stride: int = 1,
        enable_voronoi: Optional[bool] = None,
        collect_stats: bool = False,
        stats_dir: str = "reports",
        stats_config: Optional[MatchStatsConfig] = None,
        enable_overlay: bool = False,
    ) -> None:
        """
        Process an input video and write the resulting annotated video with mini-map to disk.
        Fully supports both .mp4 and .webm containers.
        """
        source_p = Path(source_path).resolve()
        target_p = Path(target_path).resolve()

        # Prevent overwriting source video
        if source_p == target_p:
            suffix = source_p.suffix if source_p.suffix else ".mp4"
            new_target = source_p.with_name(f"{source_p.stem}_annotated{suffix}")
            print(f"[Warning] Output path is identical to input video! Automatically changed to: {new_target.name}")
            target_path = str(new_target)
            target_p = new_target

        # Determine codec based on output file extension (.webm -> VP80, .mp4 -> mp4v)
        codec = get_video_codec(target_path)

        # Ensure team classifier is fitted
        if not self.team_classifier.is_fitted:
            self.fit_teams_from_video(source_path)

        self.tracker.reset()
        if hasattr(self.pitch_detector, "_auto_disabled"):
            self.pitch_detector._auto_disabled = False
            self.pitch_detector._consecutive_timeouts = 0
            self.pitch_detector._total_timeouts = 0

        video_info = sv.VideoInfo.from_video_path(source_path)

        # Handle abnormal FPS from Ubuntu GNOME screen recordings (e.g. 1000 FPS VFR container)
        effective_stride = stride
        if stride == 1 and video_info.fps > 60:
            suggested_stride = max(1, int(round(video_info.fps / 30.0)))
            print(f"[Tối ưu FPS] Phát hiện video có FPS bất thường: {video_info.fps:.0f} FPS (đặc trưng video quay màn hình Ubuntu).")
            print(f"-> Tự động điều chỉnh stride={suggested_stride} để xuất video chuẩn 30 FPS ({video_info.total_frames // suggested_stride} frames thay vì {video_info.total_frames} frames).")
            effective_stride = suggested_stride

        frame_generator = sv.get_video_frames_generator(source_path, stride=effective_stride)

        total_frames = video_info.total_frames // effective_stride
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        output_fps = video_info.fps / effective_stride
        if output_fps > 60:
            output_fps = 30.0

        output_video_info = sv.VideoInfo(
            width=video_info.width,
            height=video_info.height,
            fps=output_fps,
            total_frames=total_frames,
        )

        target_dir = Path(target_path).parent
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting video processing: {source_path} -> {target_path}")
        print(f"Resolution: {video_info.width}x{video_info.height}, FPS: {output_fps:.1f}, Frames: {total_frames}, Codec: {codec}")

        with sv.VideoSink(target_path=target_path, video_info=output_video_info, codec=codec) as sink:
            stats_recorder = MatchStatsRecorder(config=stats_config) if collect_stats else None
            live_overlay = LiveStatsOverlay(config=stats_config) if enable_overlay else None
            if live_overlay is not None:
                live_overlay.reset()
            for i, frame in enumerate(tqdm(frame_generator, total=total_frames, desc="Processing frames")):
                # Real video time of this processed frame (seconds)
                frame_time = (i * effective_stride) / video_info.fps
                processed_frame = self.process_frame(
                    frame,
                    enable_voronoi=enable_voronoi,
                    stats_recorder=stats_recorder,
                    frame_time=frame_time,
                    live_overlay=live_overlay,
                )
                sink.write_frame(processed_frame)

                if max_frames is not None and i + 1 >= max_frames:
                    break

        print(f"\n[Success] Video processing completed! Saved to: {target_path}")

        # Ensure web compatibility (H.264 / yuv420p) so browser can play video directly
        self._ensure_web_compatible_mp4(target_path)

        # ---- post-match statistics --------------------------------------- #
        if stats_recorder is not None and len(stats_recorder) > 0:
            analyzer = MatchStatsAnalyzer(
                frames=stats_recorder.frames,
                pitch_config=self.pitch_config,
                config=stats_config or stats_recorder.config,
                video_meta={
                    "file": Path(source_path).name,
                    "container_fps": video_info.fps,
                    "stride": effective_stride,
                    "processed_frames": i + 1,
                },
            )
            stats = analyzer.analyze()
            files = MatchStatsExporter.export(
                stats, out_dir=stats_dir, basename="match_stats"
            )
            self.last_match_stats = stats
            self._print_stats_summary(stats, files)

    def _print_stats_summary(self, stats: dict, files: dict) -> None:
        """Print a compact summary of the computed match statistics."""
        poss = stats["possession"]
        teams = stats["teams"]
        p = stats["processed"]
        print("\n" + "=" * 62)
        print("📊 THỐNG KÊ TRẬN ĐẤU (ước lượng từ Computer Vision)")
        print("=" * 62)
        print(f"Video:     {stats['video'].get('file', '?')} | "
              f"{p['frames']} frames xử lý | {p['duration_s']:.1f}s")
        print(f"Chất lượng toạ độ: {p.get('coordinate_quality', '?')} "
              f"(wobble ~{p.get('coordinate_wobble_est_ms', '?')} m/s)")
        print(f"Cầm bóng:  Đội 0 = {poss['possession_share_pct'].get(0, 0):.1f}% | "
              f"Đội 1 = {poss['possession_share_pct'].get(1, 0):.1f}%")
        for t in ("0", "1"):
            d = teams.get(t, {})
            print(f"  Đội {t}: quãng đường {d.get('total_distance_m', 0):.0f} m | "
                  f"bứt tốc {d.get('total_sprints', 0)} lần | "
                  f"tốc độ max TB {d.get('avg_top_speed_kmh', 0):.1f} km/h")
        print(f"Chuỗi cầm bóng dài nhất: Đội 0 = {poss['longest_control_run_s'].get('0', 0)}s | "
              f"Đội 1 = {poss['longest_control_run_s'].get('1', 0)}s")
        print("-" * 62)
        for name, pth in files.items():
            print(f"  ✔ {name}: {Path(pth)}")
        print("=" * 62 + "\n")

    @staticmethod
    def _ensure_web_compatible_mp4(target_path: str) -> None:
        """Convert video to H.264 (yuv420p + faststart) for 100% web browser compatibility."""
        target_p = Path(target_path)
        if target_p.suffix.lower() != ".mp4":
            return
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            temp_out = target_p.with_name(f"{target_p.stem}_web.mp4")
            cmd = [
                ffmpeg_exe, "-y",
                "-i", str(target_p),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(temp_out),
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and temp_out.exists() and temp_out.stat().st_size > 0:
                temp_out.replace(target_p)
                print(f"[Web Compatibility] Video converted to H.264 for instant browser playback: {target_p.name}")
        except Exception:
            pass
