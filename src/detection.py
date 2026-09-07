from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import supervision as sv
from ultralytics import YOLO

from .config import ModelConfig, ClassIDConfig, VisualConfig


@dataclass
class DetectionResult:
    """Container for processed detections on a single video frame."""
    ball_detections: sv.Detections
    tracked_detections: sv.Detections
    player_detections: sv.Detections
    goalkeeper_detections: sv.Detections
    referee_detections: sv.Detections


class PlayerDetector:
    """
    Object detector for football match entities (ball, goalkeeper, player, referee).
    Uses a fine-tuned YOLO model.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        class_config: Optional[ClassIDConfig] = None,
        visual_config: Optional[VisualConfig] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.class_config = class_config or ClassIDConfig()
        self.visual_config = visual_config or VisualConfig()

        # Load YOLO model
        self.model = YOLO(self.model_config.yolo_weights_path)

    def detect_raw(self, frame: np.ndarray) -> sv.Detections:
        """Run raw inference and return supervision Detections."""
        result = self.model(frame, conf=self.model_config.yolo_conf, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    def detect(self, frame: np.ndarray) -> Tuple[sv.Detections, sv.Detections]:
        """
        Run inference, isolate ball detections, and filter non-ball detections with NMS.

        Returns:
            Tuple[sv.Detections, sv.Detections]: (ball_detections, non_ball_detections)
        """
        detections = self.detect_raw(frame)

        # 1. Ball detections (padded)
        ball_mask = detections.class_id == self.class_config.ball_id
        ball_detections = detections[ball_mask]
        if len(ball_detections) > 0:
            ball_detections.xyxy = sv.pad_boxes(
                xyxy=ball_detections.xyxy,
                px=self.visual_config.ball_box_pad_px
            )

        # 2. Non-ball detections (filtered with class-agnostic NMS)
        non_ball_detections = detections[~ball_mask]
        if len(non_ball_detections) > 0:
            non_ball_detections = non_ball_detections.with_nms(
                threshold=self.model_config.yolo_iou,
                class_agnostic=True
            )

        return ball_detections, non_ball_detections


class PlayerTracker:
    """
    Multi-object tracker using ByteTrack for players, goalkeepers, and referees.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self.reset()

    def reset(self) -> None:
        """Reset internal tracker state."""
        self.tracker.reset()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with detections and assign tracker_id."""
        if len(detections) == 0:
            return detections
        return self.tracker.update_with_detections(detections=detections)
