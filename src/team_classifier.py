from typing import Generator, Iterable, List, Optional, TypeVar
import numpy as np
import supervision as sv
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from .config import ModelConfig, ClassIDConfig

V = TypeVar("V")


def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    """Generate batches from a sequence with a specified batch size."""
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Assign each goalkeeper to the nearest team centroid based on bottom-center position.

    Args:
        players: Detections of players with classified team IDs (0 and 1).
        goalkeepers: Detections of goalkeepers.

    Returns:
        np.ndarray: Team IDs for the goalkeepers (0 or 1).
    """
    if len(goalkeepers) == 0:
        return np.array([], dtype=int)

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_points = players_xy[players.class_id == 0]
    team_1_points = players_xy[players.class_id == 1]

    if len(team_0_points) == 0 and len(team_1_points) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
    if len(team_0_points) == 0:
        return np.ones(len(goalkeepers), dtype=int)
    if len(team_1_points) == 0:
        return np.zeros(len(goalkeepers), dtype=int)

    team_0_centroid = team_0_points.mean(axis=0)
    team_1_centroid = team_1_points.mean(axis=0)

    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id, dtype=int)


def collect_player_crops(
    video_path: str,
    detector,
    player_id: int = 2,
    stride: int = 30,
    max_crops: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Sample video frames with a given stride and extract crops of detected players.
    """
    frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=stride)
    crops: List[np.ndarray] = []

    for frame in tqdm(frame_generator, desc="Collecting player crops for team clustering"):
        _, non_ball_detections = detector.detect(frame)
        players = non_ball_detections[non_ball_detections.class_id == player_id]
        frame_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
        crops.extend(frame_crops)

        if max_crops is not None and len(crops) >= max_crops:
            crops = crops[:max_crops]
            break

    return crops


class TeamClassifier:
    """
    Clusters player crops into 2 teams using SigLIP embeddings and KMeans.
    Reduces dimensions with UMAP (or PCA fallback) before clustering.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 32,
        model_path: str = "google/siglip-base-patch16-224",
        use_umap: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_path = model_path
        self.use_umap = use_umap

        self._features_model = None
        self._processor = None
        self._reducer = None
        self._cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.is_fitted = False

    def _init_models(self):
        if self._features_model is None:
            from transformers import AutoProcessor, SiglipVisionModel
            self._features_model = SiglipVisionModel.from_pretrained(self.model_path).to(self.device)
            self._processor = AutoProcessor.from_pretrained(self.model_path)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract mean pooled embeddings from image crops using SigLIP."""
        if len(crops) == 0:
            return np.empty((0, 768))

        self._init_models()
        pil_crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(pil_crops, self.batch_size)
        data = []

        with torch.no_grad():
            for batch in tqdm(batches, desc="Embedding extraction", leave=False):
                inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self._features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data, axis=0)

    def _fit_reducer(self, data: np.ndarray) -> np.ndarray:
        n_samples = len(data)
        n_components = min(3, n_samples)

        if self.use_umap and n_samples >= 15:
            try:
                import umap
                self._reducer = umap.UMAP(n_components=n_components, random_state=42)
                return self._reducer.fit_transform(data)
            except Exception:
                pass

        self._reducer = PCA(n_components=n_components, random_state=42)
        return self._reducer.fit_transform(data)

    def fit(self, crops: List[np.ndarray]) -> "TeamClassifier":
        """Fit dimensionality reducer and KMeans on player image crops."""
        if len(crops) < 2:
            raise ValueError(f"Need at least 2 crops to fit team classifier, got {len(crops)}")

        self._init_models()
        data = self.extract_features(crops)
        projections = self._fit_reducer(data)
        self._cluster_model.fit(projections)
        self.is_fitted = True
        return self

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Predict team IDs (0 or 1) for player image crops."""
        if len(crops) == 0:
            return np.array([], dtype=int)

        if not self.is_fitted:
            raise RuntimeError("TeamClassifier must be fitted before predict() is called.")

        data = self.extract_features(crops)
        projections = self._reducer.transform(data)
        return self._cluster_model.predict(projections)

    def get_projections(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract features and project to 3D for visualization."""
        self._init_models()
        data = self.extract_features(crops)
        if not self.is_fitted:
            return self._fit_reducer(data)
        return self._reducer.transform(data)
