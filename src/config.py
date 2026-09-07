import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch


def download_file(url: str, dest: Path):
    """Download a file safely with temporary file to prevent corrupted partial files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_file = dest.with_suffix(dest.suffix + ".part")
    print(f"[Auto-Download] Downloading {dest.name} from {url}...")
    import urllib.request
    urllib.request.urlretrieve(url, part_file)
    if part_file.exists() and part_file.stat().st_size > 0:
        if dest.exists():
            dest.unlink()
        part_file.rename(dest)
        print(f"[Auto-Download] Successfully downloaded {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)!")


def find_default_weights() -> str:
    """Find local weights file if available, or auto-download from GitHub release."""
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "football/runs/detect/train/weights/best.pt",
        project_root / "best.pt",
        project_root / "football/yolov8m.pt",
        project_root / "yolov8m.pt",
        Path("football/runs/detect/train/weights/best.pt"),
        Path("best.pt"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1000:
            return str(p.resolve())

    # If no local weights exist, auto-download fine-tuned best.pt from GitHub Release
    target = project_root / "football/runs/detect/train/weights/best.pt"
    release_url = "https://github.com/ducanhdhtb06-hub/football-match-analysis/releases/download/v1.0.0/best.pt"
    try:
        download_file(release_url, target)
        if target.exists() and target.stat().st_size > 1000:
            return str(target.resolve())
    except Exception as e:
        print(f"[Auto-Download Warning] Could not download weights from {release_url}: {e}")

    return str(target)


@dataclass
class ModelConfig:
    """Configuration for AI models and detection thresholds."""
    yolo_weights_path: str = field(default_factory=find_default_weights)
    yolo_conf: float = 0.3
    yolo_iou: float = 0.5

    # Roboflow Field Detection Model
    roboflow_api_key: str = field(
        default_factory=lambda: os.environ.get("ROBOFLOW_API_KEY", "UxApJs5oZUkmngdc3qdV")
    )
    field_model_id: str = "football-field-detection-f07vi/14"
    field_conf: float = 0.3
    field_kp_conf_threshold: float = 0.5

    # SigLIP vision embedding model for team clustering
    siglip_model_path: str = "google/siglip-base-patch16-224"
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class ClassIDConfig:
    """Class IDs matching the dataset (football-players-detection-4)."""
    ball_id: int = 0
    goalkeeper_id: int = 1
    player_id: int = 2
    referee_id: int = 3


@dataclass
class VisualConfig:
    """Visualization colors, line widths, and radar/mini-map settings."""
    # Team & referee colors
    team_0_hex: str = "#00BFFF"  # Deep Sky Blue
    team_1_hex: str = "#FF1493"  # Deep Pink
    referee_hex: str = "#FFD700"  # Gold
    ball_hex: str = "#FFD700"  # Gold marker on video

    # Pitch radar colors
    pitch_background_rgb: tuple = (34, 139, 34)  # Forest Green
    pitch_line_color_hex: str = "#FFFFFF"
    ball_pitch_face_hex: str = "#FFFFFF"
    ball_pitch_edge_hex: str = "#000000"

    # Annotator parameters
    ellipse_thickness: int = 2
    triangle_base: int = 20
    triangle_height: int = 17
    ball_box_pad_px: int = 10

    # Mini-map / radar overlay parameters
    radar_scale: float = 0.1
    radar_padding: int = 50
    mini_map_scale: float = 0.5
    mini_map_margin: int = 10
    mini_map_position: str = "bottom-right"  # 'bottom-right', 'top-right', 'bottom-left', 'top-left'

    # Voronoi diagram parameters
    voronoi_opacity: float = 0.5
    voronoi_steepness: float = 15.0


@dataclass
class PipelineConfig:
    """Pipeline execution parameters."""
    crop_stride: int = 30  # Frame stride to collect player crops for team classifier fitting
    max_fitting_crops: Optional[int] = None
    batch_size: int = 32
    enable_voronoi: bool = False
    cache_homography: bool = True  # Reuse last known valid homography if current frame has <4 keypoints
    # Temporal smoothing of the homography matrix (per-frame keypoint noise).
    smooth_homography: bool = True
    homography_smoothing_alpha: float = 0.15   # lower = smoother (0 < a <= 1)
    # Reject single-frame homography jumps larger than this (pitch cm) unless
    # the view genuinely changed for many consecutive frames (re-baseline).
    homography_stickiness_cm: float = 500.0
    homography_rebaseline_frames: int = 12
    # Ignore tracked entities whose bounding box is shorter than this (px).
    # Removes tiny far-field false detections that destabilize track ids.
    min_entity_height_px: float = 16.0
    # Run pitch keypoint (Roboflow) inference every N processed frames and
    # reuse the result otherwise (1 = every frame).  Big speed-up when the
    # Roboflow API is slow; homography stays cached between calls.
    pitch_keypoint_every: int = 1
