import argparse
from pathlib import Path
import cv2
import numpy as np
import supervision as sv

from src.config import ModelConfig
from src.pitch_config import SoccerPitchConfiguration
from src.pitch_detector import PitchKeypointDetector
from src.projection import ViewTransformer


def test_pitch_reprojection(
    video_path: str,
    frame_index: int = 200,
    api_key: str = None,
    output_image_path: str = "reports/pitch_homography_test.jpg"
):
    """
    Test pitch keypoint detection and reproject pitch lines onto camera frame.
    (Modularized from Cell 28-31 of the notebook).
    """
    model_config = ModelConfig()
    if api_key:
        model_config.roboflow_api_key = api_key

    pitch_config = SoccerPitchConfiguration()
    pitch_detector = PitchKeypointDetector(model_config=model_config, pitch_config=pitch_config)

    # Load specific frame with fallback to sample frames if pitch is not clearly visible
    frame = None
    frame_ref_points = np.empty((0, 2), dtype=np.float32)
    pitch_ref_points = np.empty((0, 2), dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidate_indices = [frame_index, 100, 150, 80, 50, 250, 300, 10, 0]
    candidate_indices = [idx for idx in dict.fromkeys(candidate_indices) if total_frames <= 0 or idx < total_frames]

    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, cand_frame = cap.read()
        if not ret or cand_frame is None:
            continue
        key_points = pitch_detector.detect(cand_frame)
        f_ref, p_ref, _ = pitch_detector.get_reference_points(key_points)
        if len(f_ref) >= 4:
            frame = cand_frame
            frame_ref_points = f_ref
            pitch_ref_points = p_ref
            print(f"Detected {len(frame_ref_points)} valid pitch keypoints on frame {idx} (confidence > {model_config.field_kp_conf_threshold})")
            break
        elif frame is None:
            frame = cand_frame
            frame_ref_points = f_ref
            pitch_ref_points = p_ref
    cap.release()

    if len(frame_ref_points) < 4:
        print("Warning: Fewer than 4 keypoints found in sampled frames, cannot compute homography.")
        return

    # 2. Filter inliers using RANSAC to eliminate false keypoint detections
    m, inliers = cv2.findHomography(pitch_ref_points, frame_ref_points, cv2.RANSAC, 15.0)
    if inliers is not None and np.sum(inliers) >= 4:
        inlier_mask = (inliers.ravel() == 1)
        pitch_ref_points = pitch_ref_points[inlier_mask]
        frame_ref_points = frame_ref_points[inlier_mask]

    # Compute refined Homography: Pitch model -> Camera frame
    transformer = ViewTransformer(source=pitch_ref_points, target=frame_ref_points)

    # Transform all pitch vertices to frame
    pitch_all_points = np.array(pitch_config.vertices, dtype=np.float32)
    frame_all_points = transformer.transform_points(pitch_all_points)

    frame_all_kp = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
    frame_detected_kp = sv.KeyPoints(xy=frame_ref_points[np.newaxis, ...])

    # Annotators
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.from_hex("#00BFFF"),
        thickness=2,
        edges=pitch_config.edges
    )
    detected_kp_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex("#FF1493"),
        radius=8
    )
    reprojected_kp_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex("#00BFFF"),
        radius=6
    )

    annotated = frame.copy()
    # Draw reprojected field pitch edges
    annotated = edge_annotator.annotate(scene=annotated, key_points=frame_all_kp)
    # Draw reprojected pitch vertices (cyan)
    annotated = reprojected_kp_annotator.annotate(scene=annotated, key_points=frame_all_kp)
    # Draw originally detected keypoints (pink)
    annotated = detected_kp_annotator.annotate(scene=annotated, key_points=frame_detected_kp)

    out_path = Path(output_image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"[Success] Reprojected pitch test frame saved to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Test pitch keypoint detection and homography alignment.")
    parser.add_argument("--video", type=str, default="football/test.mp4", help="Video file path")
    parser.add_argument("--frame-index", type=int, default=200, help="Frame index to test")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API key")
    parser.add_argument("--output", type=str, default="reports/pitch_homography_test.jpg", help="Output image file")
    args = parser.parse_args()

    test_pitch_reprojection(
        video_path=args.video,
        frame_index=args.frame_index,
        api_key=args.api_key,
        output_image_path=args.output,
    )


if __name__ == "__main__":
    main()
