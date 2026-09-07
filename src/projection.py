from typing import Optional, Tuple
import cv2
import numpy as np
import supervision as sv


class ViewTransformer:
    """
    Computes and applies 2D perspective (homography) transformation between planes.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Args:
            source: Source 2D points (N, 2).
            target: Target 2D points (N, 2).
        """
        if len(source) < 4 or len(target) < 4:
            raise ValueError(f"Need at least 4 point correspondences for homography, got {len(source)}")

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.m, self.inliers = cv2.findHomography(source, target, cv2.RANSAC, 15.0)
        if self.m is None:
            raise ValueError("Homography matrix could not be computed.")
        if self.inliers is not None and np.sum(self.inliers) >= 4:
            inlier_mask = (self.inliers.ravel() == 1)
            refined_m, _ = cv2.findHomography(source[inlier_mask], target[inlier_mask])
            if refined_m is not None:
                self.m = refined_m

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform 2D points using homography matrix."""
        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed.reshape(-1, 2)


class PitchProjector:
    """
    Manages frame-to-pitch coordinate projections across video frames.
    Includes temporal caching of homography to ensure stability during occlusions,
    plus optional exponential smoothing of the homography matrix to damp
    frame-to-frame keypoint noise that would otherwise jitter every projected
    coordinate (and corrupt speed/distance metrics).
    """

    def __init__(
        self,
        cache_homography: bool = True,
        smooth_homography: bool = True,
        homography_smoothing_alpha: float = 0.35,
        stickiness_cm: float = 500.0,
        rebaseline_frames: int = 12,
    ):
        self.cache_homography = cache_homography
        self.smooth_homography = smooth_homography
        self.smoothing_alpha = homography_smoothing_alpha
        self.stickiness_cm = float(stickiness_cm)
        self.rebaseline_frames = int(rebaseline_frames)
        self.last_valid_transformer: Optional[ViewTransformer] = None
        self._smoothed_m: Optional[np.ndarray] = None
        self._reject_streak = 0
        self._pending_m: Optional[np.ndarray] = None   # first candidate H (not yet trusted)

    @staticmethod
    def _projected_center_drift(
        m_ref: np.ndarray, m_cand: np.ndarray,
        pts_cm: np.ndarray,
    ) -> float:
        """Median displacement (cm) of reference points projected by both Hs."""
        def apply(m, pts):
            p = pts.reshape(-1, 1, 2).astype(np.float32)
            out = cv2.perspectiveTransform(p, m).reshape(-1, 2)
            return out
        a = apply(m_ref, pts_cm)
        b = apply(m_cand, pts_cm)
        d = np.linalg.norm(a - b, axis=1)
        return float(np.median(d))

    def update(
        self,
        frame_points: np.ndarray,
        pitch_points: np.ndarray
    ) -> Optional[ViewTransformer]:
        """
        Update the projection matrix with newly detected keypoints.
        Returns the active ViewTransformer, or None if no valid homography exists yet.
        """
        if len(frame_points) >= 4 and len(pitch_points) >= 4:
            try:
                transformer = ViewTransformer(source=frame_points, target=pitch_points)
                m_raw = transformer.m
                if self.smooth_homography:
                    if self._smoothed_m is None:
                        # ---- safe initialisation ------------------------------ #
                        # A single noisy fit (wrong keypoint matches) can be
                        # self-consistent but geometrically wrong -> possession 0.
                        # Lock the reference only when TWO consecutive fits agree.
                        probe = np.array([[0.0, 0.0], [105.0, 0.0], [0.0, 68.0], [105.0, 68.0]],
                                         dtype=np.float32) * 100.0
                        if self._pending_m is None:
                            self._pending_m = m_raw.copy()
                            return None
                        agree = self._projected_center_drift(
                            self._pending_m, m_raw, probe) <= self.stickiness_cm
                        if not agree:
                            self._pending_m = m_raw.copy()   # keep searching
                            return None
                        self._smoothed_m = m_raw.copy()
                        self._pending_m = None
                    else:
                        # ---- stickiness: reject single-frame jumps ---------- #
                        # If the fresh homography would move the pitch template
                        # far from the current smoothed one, it is a noisy
                        # re-detection -> ignore it (keep the smoothed state).
                        probe = np.array(
                            [[0.0, 0.0], [105.0, 0.0], [0.0, 68.0], [105.0, 68.0]],
                            dtype=np.float32,
                        ) * 100.0   # meters -> pitch cm
                        drift = self._projected_center_drift(
                            self._smoothed_m, m_raw, probe)
                        if drift <= self.stickiness_cm:
                            a = float(self.smoothing_alpha)
                            self._smoothed_m = a * m_raw + (1.0 - a) * self._smoothed_m
                            self._reject_streak = 0
                        else:
                            self._reject_streak += 1
                            if self._reject_streak >= self.rebaseline_frames:
                                # view genuinely changed (cut / camera move):
                                # re-baseline to the fresh homography
                                self._smoothed_m = m_raw.copy()
                                self._reject_streak = 0
                    transformer.m = self._smoothed_m.copy()
                self.last_valid_transformer = transformer
                return transformer
            except Exception:
                pass

        if self.cache_homography and self.last_valid_transformer is not None:
            return self.last_valid_transformer

        return None

    def project_detections(
        self,
        transformer: Optional[ViewTransformer],
        detections: sv.Detections,
        anchor: sv.Position = sv.Position.BOTTOM_CENTER
    ) -> np.ndarray:
        """
        Extract anchor coordinates (bottom-center by default) from detections
        and project them onto pitch coordinates.
        """
        if transformer is None or len(detections) == 0:
            return np.empty((0, 2), dtype=np.float32)

        frame_xy = detections.get_anchors_coordinates(anchor)
        return transformer.transform_points(frame_xy)

    def project_points(
        self,
        transformer: Optional[ViewTransformer],
        points: np.ndarray
    ) -> np.ndarray:
        """Project raw (N, 2) points onto pitch coordinates."""
        if transformer is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        return transformer.transform_points(points)
