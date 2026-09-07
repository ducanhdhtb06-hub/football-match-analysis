from typing import Optional
import cv2
import numpy as np
import supervision as sv

from .config import VisualConfig
from .pitch_config import SoccerPitchConfiguration


def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1,
) -> np.ndarray:
    """
    Draws a 2D bird's-eye view soccer pitch with lines and penalty marks.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # Draw boundary and penalty lines
    for start, end in config.edges:
        point1 = (
            int(config.vertices[start - 1][0] * scale) + padding,
            int(config.vertices[start - 1][1] * scale) + padding,
        )
        point2 = (
            int(config.vertices[end - 1][0] * scale) + padding,
            int(config.vertices[end - 1][1] * scale) + padding,
        )
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness,
        )

    # Centre circle
    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding,
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness,
    )

    # Penalty spots
    penalty_spots = [
        (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
        (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
        centre_circle_center,
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1,
        )

    return pitch_image


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw circular markers on the 2D pitch representing players, ball, or referees."""
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    if xy is None or len(xy) == 0:
        return pitch

    for point in xy:
        # Ignore out-of-bounds or NaN points
        if np.isnan(point).any():
            continue
        scaled_x = int(point[0] * scale) + padding
        scaled_y = int(point[1] * scale) + padding

        # Ensure point is within pitch image bounds
        if 0 <= scaled_x < pitch.shape[1] and 0 <= scaled_y < pitch.shape[0]:
            cv2.circle(
                img=pitch,
                center=(scaled_x, scaled_y),
                radius=radius,
                color=face_color.as_bgr(),
                thickness=-1,
            )
            cv2.circle(
                img=pitch,
                center=(scaled_x, scaled_y),
                radius=radius,
                color=edge_color.as_bgr(),
                thickness=thickness,
            )

    return pitch


def draw_pitch_voronoi(
    config: SoccerPitchConfiguration,
    team_0_xy: np.ndarray,
    team_1_xy: np.ndarray,
    team_0_color: sv.Color = sv.Color.from_hex("#00BFFF"),
    team_1_color: sv.Color = sv.Color.from_hex("#FF1493"),
    opacity: float = 0.5,
    steepness: float = 15.0,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draws a Voronoi diagram on a soccer pitch representing the control areas of two
    teams with smooth color transitions using tanh blending.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    if len(team_0_xy) == 0 or len(team_1_xy) == 0:
        return pitch

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_0_color_bgr = np.array(team_0_color.as_bgr(), dtype=np.uint8)
    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)

    y_coords, x_coords = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding,
    ))
    y_coords = y_coords - padding
    x_coords = x_coords - padding

    def calc_distances(xy, xc, yc):
        return np.sqrt((xy[:, 0][:, None, None] * scale - xc) ** 2 +
                       (xy[:, 1][:, None, None] * scale - yc) ** 2)

    dist_team_0 = calc_distances(team_0_xy, x_coords, y_coords)
    dist_team_1 = calc_distances(team_1_xy, x_coords, y_coords)

    min_dist_team_0 = np.min(dist_team_0, axis=0)
    min_dist_team_1 = np.min(dist_team_1, axis=0)

    dist_ratio = min_dist_team_1 / np.clip(min_dist_team_0 + min_dist_team_1, a_min=1e-5, a_max=None)
    blend = np.tanh((dist_ratio - 0.5) * steepness) * 0.5 + 0.5

    for c in range(3):
        voronoi[:, :, c] = (blend * team_0_color_bgr[c] + (1 - blend) * team_1_color_bgr[c]).astype(np.uint8)

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)
    return overlay


class PitchDrawer:
    """Helper class managing the complete 2D pitch radar visualization and overlay."""

    def __init__(
        self,
        config: Optional[SoccerPitchConfiguration] = None,
        visual_config: Optional[VisualConfig] = None,
    ):
        self.config = config or SoccerPitchConfiguration()
        self.visual_config = visual_config or VisualConfig()

    def build_radar_view(
        self,
        ball_xy: np.ndarray,
        team_0_xy: np.ndarray,
        team_1_xy: np.ndarray,
        referee_xy: np.ndarray,
        enable_voronoi: bool = False,
    ) -> np.ndarray:
        """Create the complete 2D radar view image with pitch lines, players, ball, and referee."""
        pitch = draw_pitch(
            config=self.config,
            background_color=sv.Color(*self.visual_config.pitch_background_rgb),
            line_color=sv.Color.from_hex(self.visual_config.pitch_line_color_hex),
            padding=self.visual_config.radar_padding,
            scale=self.visual_config.radar_scale,
        )

        if enable_voronoi and len(team_0_xy) > 0 and len(team_1_xy) > 0:
            pitch = draw_pitch_voronoi(
                config=self.config,
                team_0_xy=team_0_xy,
                team_1_xy=team_1_xy,
                team_0_color=sv.Color.from_hex(self.visual_config.team_0_hex),
                team_1_color=sv.Color.from_hex(self.visual_config.team_1_hex),
                opacity=self.visual_config.voronoi_opacity,
                steepness=self.visual_config.voronoi_steepness,
                padding=self.visual_config.radar_padding,
                scale=self.visual_config.radar_scale,
                pitch=pitch,
            )

        # 1. Draw Team 0 (cyan)
        pitch = draw_points_on_pitch(
            config=self.config,
            xy=team_0_xy,
            face_color=sv.Color.from_hex(self.visual_config.team_0_hex),
            edge_color=sv.Color.BLACK,
            radius=16,
            padding=self.visual_config.radar_padding,
            scale=self.visual_config.radar_scale,
            pitch=pitch,
        )

        # 2. Draw Team 1 (pink)
        pitch = draw_points_on_pitch(
            config=self.config,
            xy=team_1_xy,
            face_color=sv.Color.from_hex(self.visual_config.team_1_hex),
            edge_color=sv.Color.BLACK,
            radius=16,
            padding=self.visual_config.radar_padding,
            scale=self.visual_config.radar_scale,
            pitch=pitch,
        )

        # 3. Draw Referee (gold)
        pitch = draw_points_on_pitch(
            config=self.config,
            xy=referee_xy,
            face_color=sv.Color.from_hex(self.visual_config.referee_hex),
            edge_color=sv.Color.BLACK,
            radius=16,
            padding=self.visual_config.radar_padding,
            scale=self.visual_config.radar_scale,
            pitch=pitch,
        )

        # 4. Draw Ball (white with black border)
        pitch = draw_points_on_pitch(
            config=self.config,
            xy=ball_xy,
            face_color=sv.Color.from_hex(self.visual_config.ball_pitch_face_hex),
            edge_color=sv.Color.from_hex(self.visual_config.ball_pitch_edge_hex),
            radius=10,
            padding=self.visual_config.radar_padding,
            scale=self.visual_config.radar_scale,
            pitch=pitch,
        )

        return pitch

    def overlay_mini_map(
        self,
        main_frame: np.ndarray,
        radar_view: np.ndarray,
    ) -> np.ndarray:
        """Resize and place the 2D radar view as a mini-map overlay on the main frame."""
        pitch_h, pitch_w, _ = radar_view.shape
        mini_w = int(pitch_w * self.visual_config.mini_map_scale)
        mini_h = int(pitch_h * self.visual_config.mini_map_scale)

        mini_map = cv2.resize(radar_view, (mini_w, mini_h))
        main_h, main_w, _ = main_frame.shape
        margin = self.visual_config.mini_map_margin
        pos = self.visual_config.mini_map_position

        if pos == "bottom-right":
            x = main_w - mini_w - margin
            y = main_h - mini_h - margin
        elif pos == "top-right":
            x = main_w - mini_w - margin
            y = margin
        elif pos == "bottom-left":
            x = margin
            y = main_h - mini_h - margin
        else:  # top-left
            x = margin
            y = margin

        result_frame = main_frame.copy()
        if 0 <= x and x + mini_w <= main_w and 0 <= y and y + mini_h <= main_h:
            result_frame[y:y + mini_h, x:x + mini_w] = mini_map

        return result_frame
