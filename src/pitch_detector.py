from typing import Optional, Tuple
import concurrent.futures
from functools import partial
import numpy as np
import supervision as sv

from .config import ModelConfig
from .pitch_config import SoccerPitchConfiguration

# Sentinel returned when the remote inference call exceeds the time budget
_REQUEST_TIMEOUT_S = 15.0
_TIMEOUT = object()


class PitchKeypointDetector:
    """
    Detects keypoints on the soccer pitch using Roboflow Inference API / SDK.
    Used for camera calibration and homography calculation.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        pitch_config: Optional[SoccerPitchConfiguration] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.pitch_config = pitch_config or SoccerPitchConfiguration()
        self._model = None
        self._use_sdk = False
        self._consecutive_timeouts = 0
        self._total_timeouts = 0
        self._auto_disabled = False
        self._init_model()

    def _init_model(self):
        """Initialize Roboflow model using inference or inference_sdk."""
        try:
            from inference import get_model
            self._model = get_model(
                model_id=self.model_config.field_model_id,
                api_key=self.model_config.roboflow_api_key,
            )
            self._use_sdk = False
            return
        except Exception:
            pass

        try:
            from inference_sdk import InferenceHTTPClient
            self._model = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=self.model_config.roboflow_api_key,
            )
            self._use_sdk = True
            return
        except Exception as e:
            print(f"[Warning] Failed to initialize field detection model: {e}")
            self._model = None
            self._use_sdk = False

    def _run_with_timeout(self, fn) -> object:
        """Run a blocking call on a worker thread with a hard time budget.

        Returns _TIMEOUT when the call does not finish in time (the caller can
        then fall back to the last cached homography instead of hanging).
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn)
            try:
                return fut.result(timeout=_REQUEST_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                ex.shutdown(wait=False, cancel_futures=True)
                return _TIMEOUT

    def detect(self, frame: np.ndarray) -> Optional[sv.KeyPoints]:
        """
        Run inference on a frame to detect pitch keypoints.
        Returns sv.KeyPoints or None if inference failed / timed out.
        """
        if self._model is None or self._auto_disabled:
            return None

        try:
            if self._use_sdk:
                if self._consecutive_timeouts >= 4 or self._total_timeouts >= 8:
                    self._auto_disabled = True
                    print("[PitchKeypointDetector] Qua nhieu request timeout -> dung goi Roboflow, dung homography cache (reset moi video)")
                    return None
                result = self._run_with_timeout(
                    partial(self._model.infer, frame, model_id=self.model_config.field_model_id)
                )
                if result is _TIMEOUT:
                    self._consecutive_timeouts += 1
                    self._total_timeouts += 1
                    print("[PitchKeypointDetector] Roboflow request timed out -> fallback to cached homography")
                    return None
                self._consecutive_timeouts = 0
            else:
                result = self._model.infer(frame, confidence=self.model_config.field_conf)[0]
            return sv.KeyPoints.from_inference(result)
        except Exception as e:
            print(f"[Warning] Keypoint inference error: {e}")
            return None

    def get_reference_points(
        self,
        key_points: Optional[sv.KeyPoints]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter keypoints by confidence threshold and match them with pitch model vertices.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - frame_reference_points: (N, 2) coords on the video frame
                - pitch_reference_points: (N, 2) coords in the pitch model
                - filter_mask: (32,) boolean mask of valid points
        """
        if key_points is None or len(key_points.xy) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

        # Support both new and old supervision keypoints confidence property
        conf = getattr(key_points, 'keypoint_confidence', None)
        if conf is None or len(conf) == 0:
            conf = getattr(key_points, 'confidence', None)

        if conf is None or len(conf) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

        filter_mask = conf[0] > self.model_config.field_kp_conf_threshold

        frame_points = key_points.xy[0][filter_mask].astype(np.float32)
        pitch_vertices = np.array(self.pitch_config.vertices, dtype=np.float32)

        if len(pitch_vertices) != len(filter_mask):
            min_len = min(len(pitch_vertices), len(filter_mask))
            filter_mask = filter_mask[:min_len]
            frame_points = key_points.xy[0][:min_len][filter_mask].astype(np.float32)
            pitch_points = pitch_vertices[:min_len][filter_mask]
        else:
            pitch_points = pitch_vertices[filter_mask]

        return frame_points, pitch_points, filter_mask
