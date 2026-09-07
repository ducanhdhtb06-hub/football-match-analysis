"""
Football Match Analysis & 2D Pitch Projection Package
Modularized from Cluster_players_into_teams_and_2D_projection.ipynb
"""

import sys
from pathlib import Path

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
