import argparse
import sys
import warnings
from pathlib import Path

# Suppress harmless deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*KeyPoints.confidence.*")

for p in [
    Path.home() / "local/lib/python3.12/dist-packages",
    Path.home() / "local/lib/python3.12/site-packages",
    Path.home() / "lib/python3.12/site-packages",
    Path("/home/anh/snap/antigravity-cli/common/local/lib/python3.12/dist-packages"),
    Path("/home/anh/snap/antigravity-cli/common/lib/python3.12/site-packages"),
    Path("/home/anh/PycharmProjects/PythonProject1/.venv/lib/python3.12/site-packages"),
]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Football Match Analysis, Team Clustering & 2D Pitch Projection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        default="football/test.mp4",
        help="Path to input video file (e.g., football/test.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="football/output.mp4",
        help="Path to output video file (.mp4)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLO fine-tuned weights (best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="YOLO detection confidence threshold",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process (useful for quick testing)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for video output (e.g., 1 for full video, 3 for 1/3 frames)",
    )
    parser.add_argument(
        "--voronoi",
        action="store_true",
        help="Enable smooth Voronoi diagram on 2D pitch mini-map",
    )
    parser.add_argument(
        "--mini-map-pos",
        type=str,
        default="bottom-right",
        choices=["bottom-right", "top-right", "bottom-left", "top-left"],
        help="Position of radar mini-map overlay",
    )
    parser.add_argument(
        "--test-frame",
        type=str,
        default=None,
        help="Process a single frame and save as image (e.g., reports/test_frame.jpg)",
    )
    parser.add_argument(
        "--visualize-clusters",
        action="store_true",
        help="Run 3D interactive player jersey cluster visualization tool",
    )
    parser.add_argument(
        "--test-homography",
        action="store_true",
        help="Run pitch homography reprojection test tool",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run ALL features in one command: pitch test, 3D cluster visualizer, test frame preview, and full video analysis with Voronoi",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute post-match statistics (possession share, sprints, top speed, distance, heat-maps) and save JSON/CSV/HTML reports",
    )
    parser.add_argument(
        "--kp-every",
        type=int,
        default=None,
        help="Run pitch keypoint inference every N frames (default: every frame; use 3-5 to speed up)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Draw live overlay on the output video: possession bar + per-player speed (km/h), sprint highlighted",
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="reports",
        help="Output directory for match statistics reports (match_stats.json/.csv/.html)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check input video with smart path discovery first
    video_path = Path(args.video)
    if not video_path.exists():
        candidates = [
            Path(video_path.name),
            Path(".") / video_path.name,
            Path("football") / video_path.name,
            Path("..") / "football" / video_path.name,
            Path("/home/anh/PycharmProjects/PythonProject1/football") / video_path.name,
            Path("/home/anh/PycharmProjects/PythonProject1") / video_path.name,
            Path("/home/anh/Downloads/football") / video_path.name,
        ]
        found = False
        for c in candidates:
            if c.exists():
                video_path = c
                found = True
                break

        if not found:
            # Search project for all valid videos
            available_videos = []
            for root_dir in [Path("."), Path("football"), Path("/home/anh/Downloads/football")]:
                if root_dir.exists():
                    for ext in ["*.mp4", "*.webm", "*.mkv"]:
                        available_videos.extend(root_dir.glob(ext))
            available_videos = sorted(list(set([str(v) for v in available_videos if v.is_file()])))

            print(f"Error: Input video '{args.video}' does not exist.")
            if available_videos:
                print("\n[Gợi ý] Các video có sẵn trong dự án bạn có thể dùng:")
                for v in available_videos:
                    print(f"  - {v}")
            sys.exit(1)

    # Handle quick subtools
    if args.visualize_clusters:
        from tools.cluster_visualizer_3d import main as run_clusters
        sys.argv = [sys.argv[0], "--video", str(video_path)]
        if args.weights:
            sys.argv.extend(["--weights", args.weights])
        run_clusters()
        return

    if args.test_homography:
        from tools.test_pitch_homography import main as run_homography
        sys.argv = [sys.argv[0], "--video", str(video_path)]
        run_homography()
        return

    import cv2
    from src.config import ModelConfig, VisualConfig, PipelineConfig
    from src.pipeline import FootballAnalysisPipeline

    # Initialize configurations
    model_config = ModelConfig()
    if args.weights:
        model_config.yolo_weights_path = args.weights
    model_config.yolo_conf = args.conf

    visual_config = VisualConfig()
    visual_config.mini_map_position = args.mini_map_pos

    pipeline_config = PipelineConfig()
    pipeline_config.enable_voronoi = args.voronoi
    if args.kp_every is not None and args.kp_every > 0:
        pipeline_config.pitch_keypoint_every = args.kp_every

    # Build pipeline
    print("=" * 60)
    print("Football Match Analysis & 2D Pitch Projection Pipeline")
    print("=" * 60)
    print(f"Video:       {video_path}")
    print(f"Weights:     {model_config.yolo_weights_path}")
    print(f"Device:      {model_config.device}")
    print(f"Voronoi:     {args.voronoi}")
    print("=" * 60)

    pipeline = FootballAnalysisPipeline(
        model_config=model_config,
        visual_config=visual_config,
        pipeline_config=pipeline_config,
    )

    # All-in-one execution mode
    if args.all:
        print("\n" + "=" * 65)
        print("🚀 RUNNING ALL FOOTBALL ANALYSIS FEATURES")
        print("=" * 65)

        # 1. Pitch Homography Test
        print("\n[1/4] Testing pitch keypoint detection & homography...")
        try:
            from tools.test_pitch_homography import test_pitch_reprojection
            test_pitch_reprojection(
                video_path=str(video_path),
                output_image_path="reports/pitch_homography_test.jpg",
            )
        except Exception as e:
            print(f"[Warning] Homography test failed: {e}")

        # 2. 3D Jersey Clustering
        print("\n[2/4] Generating 3D interactive player jersey clusters...")
        try:
            from tools.cluster_visualizer_3d import (
                collect_player_crops,
                TeamClassifier,
                export_interactive_3d_html,
            )
            crops = collect_player_crops(
                video_path=str(video_path),
                detector=pipeline.detector,
                stride=20,
                max_crops=150,
            )
            if len(crops) >= 2:
                classifier = TeamClassifier(device=model_config.device)
                classifier.fit(crops)
                projections = classifier.get_projections(crops)
                labels = classifier.predict(crops)
                export_interactive_3d_html(
                    labels=labels,
                    projections=projections,
                    crops=crops,
                    output_html_path="reports/player_clusters_3d.html",
                )
        except Exception as e:
            print(f"[Warning] 3D Cluster visualizer failed: {e}")

        # 3. Single Frame Test Preview
        print("\n[3/4] Exporting single-frame test preview...")
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                if not pipeline.team_classifier.is_fitted:
                    pipeline.fit_teams_from_video(str(video_path))
                annotated = pipeline.process_frame(frame, enable_voronoi=True)
                out_preview = Path("reports/test_preview.jpg")
                out_preview.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_preview), annotated)
                print(f"[Success] Test preview saved to: {out_preview.resolve()}")
        except Exception as e:
            print(f"[Warning] Test frame export failed: {e}")

        # 4. Full Video Processing with Voronoi
        print("\n[4/4] Processing video with radar & Voronoi space diagram...")
        pipeline.process_video(
            source_path=str(video_path),
            target_path=args.output,
            max_frames=args.max_frames,
            stride=args.stride,
            enable_voronoi=True,
            collect_stats=args.stats,
            stats_dir=args.stats_dir,
            enable_overlay=args.overlay,
        )

        print("\n" + "=" * 65)
        print("✅ ALL FEATURES COMPLETED SUCCESSFULLY!")
        print("=" * 65)
        print("Summary of generated outputs:")
        print("  1. Pitch Reprojection Test:  reports/pitch_homography_test.jpg")
        print("  2. 3D Cluster Visualizer:    reports/player_clusters_3d.html")
        print("  3. Single Frame Preview:     reports/test_preview.jpg")
        print(f"  4. Video Output (Voronoi):   {args.output}")
        print("=" * 65 + "\n")
        return

    # Single frame test mode
    if args.test_frame:
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print(f"Error: Failed to read first frame from {video_path}")
            sys.exit(1)

        print("Fitting team classifier on video samples first...")
        pipeline.fit_teams_from_video(str(video_path))

        print("Processing test frame...")
        annotated = pipeline.process_frame(frame, enable_voronoi=args.voronoi)

        out_path = Path(args.test_frame)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"[Success] Test frame saved to: {out_path.resolve()}")
        return

    # Full video processing mode
    pipeline.process_video(
        source_path=str(video_path),
        target_path=args.output,
        max_frames=args.max_frames,
        stride=args.stride,
        enable_voronoi=args.voronoi,
        collect_stats=args.stats,
        stats_dir=args.stats_dir,
        enable_overlay=args.overlay,
    )


if __name__ == "__main__":
    main()
