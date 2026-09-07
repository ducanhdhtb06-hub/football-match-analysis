from typing import Optional, List
import numpy as np
import supervision as sv

from .config import VisualConfig


class VideoFrameVisualizer:
    """
    Annotates main video frames with:
    - Ellipse contours for players and referee (colored by team/referee)
    - Tracker ID labels (#ID)
    - Triangle inverted marker above the ball
    """

    def __init__(self, visual_config: Optional[VisualConfig] = None):
        self.visual_config = visual_config or VisualConfig()

        self.palette = sv.ColorPalette.from_hex([
            self.visual_config.team_0_hex,
            self.visual_config.team_1_hex,
            self.visual_config.referee_hex,
        ])

        self.ellipse_annotator = sv.EllipseAnnotator(
            color=self.palette,
            thickness=self.visual_config.ellipse_thickness,
        )

        self.label_annotator = sv.LabelAnnotator(
            color=self.palette,
            text_color=sv.Color.from_hex("#000000"),
            text_position=sv.Position.BOTTOM_CENTER,
        )

        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex(self.visual_config.ball_hex),
            base=self.visual_config.triangle_base,
            height=self.visual_config.triangle_height,
        )

    def annotate(
        self,
        frame: np.ndarray,
        all_detections: sv.Detections,
        ball_detections: sv.Detections,
    ) -> np.ndarray:
        """
        Annotate a video frame with tracked entities and ball.

        Args:
            frame: Video frame (BGR image).
            all_detections: Tracked detections with class_id 0 (Team 0), 1 (Team 1), 2 (Referee).
            ball_detections: Detections containing ball bounding boxes.

        Returns:
            np.ndarray: Annotated BGR frame.
        """
        annotated_frame = frame.copy()

        # Annotate players & referees if any
        if len(all_detections) > 0:
            labels: List[str] = []
            if all_detections.tracker_id is not None:
                labels = [f"#{tid}" if tid is not None else "" for tid in all_detections.tracker_id]
            else:
                labels = ["" for _ in range(len(all_detections))]

            # Cast class_id to int to ensure correct palette indexing
            all_detections.class_id = all_detections.class_id.astype(int)

            annotated_frame = self.ellipse_annotator.annotate(
                scene=annotated_frame,
                detections=all_detections,
            )
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=all_detections,
                labels=labels,
            )

        # Annotate ball
        if len(ball_detections) > 0:
            annotated_frame = self.triangle_annotator.annotate(
                scene=annotated_frame,
                detections=ball_detections,
            )

        return annotated_frame
