import logging
import pickle as pkl
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics import pairwise_distances

from src.dataset_utils import TempDataset, Trajectory, embedding_collate

from .measure_model import MeasureModel, ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, partial[KernelPCA]] = {
    "pca": partial(KernelPCA, kernel="linear"),
    "kpca": partial(KernelPCA, kernel="rbf"),
}


class PCAMeasure(MeasureModel):
    def _init(self, model_cfg: ModelConfig) -> None:
        model_type = model_cfg.model_type
        if model_type not in ["pca", "kpca"]:
            raise ValueError(
                f"Unknown model_type {model_type} (must be one of {['pca', 'kpca']})"
            )
        model_cls = MODEL_REGISTRY[model_type]
        self._model_fn = partial(model_cls, n_components=model_cfg.latent_dim)

    def _init_from_checkpoint(self, path: Path) -> None:
        with open(path, mode="rb") as f:
            ckpt = pkl.load(f)
            self.model_cfg = ckpt["model_cfg"]
            self.model = ckpt["model"]
            self.lb = ckpt["lb"]
            self.ub = ckpt["ub"]

    def _train(self, train_cfg: TrainingConfig) -> None:
        trajectories = TempDataset(
            dataset_dir=Path(train_cfg.training_dataset_path)
        )
        embeddings = embedding_collate(trajectories).detach().cpu().numpy()

        # gamma is only used by rbf kernel
        dists = pairwise_distances(embeddings)
        sigma2 = np.median(dists**2)
        gamma = 1.0 / (2 * sigma2)
        self.model = self._model_fn(gamma=gamma)

        Z_train = self.model.fit_transform(embeddings)
        self.lb = np.quantile(Z_train, 0.05, axis=0)
        self.ub = np.quantile(Z_train, 0.95, axis=0)

        if train_cfg.save_to is not None:
            self._save_checkpoint(Path(train_cfg.save_to))

    def _save_checkpoint(self, path: Path) -> None:
        with open(path, "wb") as f:
            pkl.dump(
                {
                    "model_cfg": self.model_cfg,
                    "model": self.model,
                    "lb": self.lb,
                    "ub": self.ub,
                },
                f,
            )

        logger.warning(f"Saved pca checkpoint to {path}")

    def compute_measures_from_embeddings(
        self, embeddings: np.ndarray
    ) -> np.ndarray:
        Z = self.model.transform(embeddings)
        # min-max normalization to 2 * [lb, ub]
        return (Z - (self.lb - 0.5 * (self.ub - self.lb))) / (
            2 * (self.ub - self.lb)
        )

    def compute_measures(self, trajectories: List[Trajectory]) -> np.ndarray:
        embeddings = embedding_collate(trajectories).detach().cpu().numpy()
        return self.compute_measures_from_embeddings(embeddings)
