"""
Football Match Analysis & 2D Pitch Projection Package
Modularized from Cluster_players_into_teams_and_2D_projection.ipynb
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Search local virtualenvs if present
for venv_name in [".venv", "venv"]:
    venv_dir = BASE_DIR / venv_name
    if venv_dir.exists():
        for sp in list(venv_dir.glob("lib/python*/site-packages")) + list(venv_dir.glob("Lib/site-packages")):
            if sp.exists() and str(sp) not in sys.path:
                sys.path.insert(0, str(sp))

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
