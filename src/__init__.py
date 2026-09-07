"""
Football Match Analysis & 2D Pitch Projection Package
Modularized from Cluster_players_into_teams_and_2D_projection.ipynb
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Search local virtual environments and user package directories
search_dirs = [
    Path.home() / "local/lib/python3.12/dist-packages",
    Path.home() / "local/lib/python3.12/site-packages",
    Path.home() / "lib/python3.12/site-packages",
    Path.home() / ".local/lib/python3.12/site-packages",
    Path.home() / ".local/lib/python3.12/dist-packages",
    Path("/home/anh/snap/antigravity-cli/common/local/lib/python3.12/dist-packages"),
    Path("/home/anh/snap/antigravity-cli/common/lib/python3.12/site-packages"),
    Path("/home/anh/.local/lib/python3.12/site-packages"),
    Path("/home/anh/.local/lib/python3.12/dist-packages"),
    Path("/home/anh/local/lib/python3.12/dist-packages"),
]

for venv_name in [".venv", "venv"]:
    venv_dir = BASE_DIR / venv_name
    if venv_dir.exists():
        search_dirs.extend(venv_dir.glob("lib/python*/site-packages"))
        search_dirs.extend(venv_dir.glob("lib/python*/dist-packages"))
        search_dirs.extend(venv_dir.glob("Lib/site-packages"))

for p in search_dirs:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from .config import (
    ModelConfig,
    ClassIDConfig,
    VisualConfig,
    PipelineConfig,
)
from .detection import PlayerDetector, PlayerTracker
from .team_classifier import (
    TeamClassifier,
    resolve_goalkeepers_team_id,
    collect_player_crops,
)
from .pitch_config import SoccerPitchConfiguration
from .pitch_detector import PitchKeypointDetector
from .projection import ViewTransformer, PitchProjector
from .pitch_annotator import PitchDrawer
from .visualizer import VideoFrameVisualizer
from .pipeline import FootballAnalysisPipeline

__all__ = [
    "ModelConfig",
    "ClassIDConfig",
    "VisualConfig",
    "PipelineConfig",
    "PlayerDetector",
    "PlayerTracker",
    "TeamClassifier",
    "resolve_goalkeepers_team_id",
    "collect_player_crops",
    "SoccerPitchConfiguration",
    "PitchKeypointDetector",
    "ViewTransformer",
    "PitchProjector",
    "PitchDrawer",
    "VideoFrameVisualizer",
    "FootballAnalysisPipeline",
]
