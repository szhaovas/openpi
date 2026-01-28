import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray

from src.dataset_utils import TempDataset, Trajectory, embedding_collate

from .measure_model import MeasureModel, ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


class SVDMeasure(MeasureModel):
    def _init(self, model_cfg: ModelConfig) -> None:
        model_type = model_cfg.model_type
        if model_type != "svd":
            raise ValueError(f"Unknown model_type {model_type} (must be svd)")
        self.latent_dim = model_cfg.latent_dim

    def _init_from_checkpoint(self, path: Path) -> None:
        svd_stats = np.load(path)
        self.latent_dim = svd_stats["latent_dim"][0]
        self.mean = svd_stats["mean"]
        self.eigval = svd_stats["eigval"]
        self.U = svd_stats["U"]
        self.S = svd_stats["S"]
        self.Vt = svd_stats["Vt"]

    def _train(self, train_cfg: TrainingConfig) -> None:
        trajectories = TempDataset(
            dataset_dir=Path(train_cfg.training_dataset_path)
        )
        embeddings = embedding_collate(trajectories).detach().cpu().numpy()

        self.mean = np.mean(embeddings, axis=0)

        self.U, self.S, self.Vt = np.linalg.svd(
            embeddings - self.mean, full_matrices=False
        )

        self.eigval = (self.S**2) / (embeddings.shape[0] - 1)

        if train_cfg.save_to is not None:
            self._save_checkpoint(Path(train_cfg.save_to))

    def _save_checkpoint(self, path: Path) -> None:
        np.savez(
            path,
            laten_dim=np.array(self.latent_dim),
            mean=self.mean,
            eigval=self.eigval,
            U=self.U,
            S=self.S,
            Vt=self.Vt,
        )
        # TODO: clean this up
        os.rename(f"{path}.npz", str(path))

        logger.warning(f"Saved svd checkpoint to {path}")

    def compute_measures(self, trajectories: List[Trajectory]) -> NDArray:
        embeddings = embedding_collate(trajectories).detach().cpu().numpy()
        Z = (
            (embeddings - self.mean)
            @ self.Vt[: self.latent_dim].T
            / np.sqrt(self.eigval[: self.latent_dim])
        )
        recon = (Z * np.sqrt(self.eigval[: self.latent_dim])) @ self.Vt[
            : self.latent_dim
        ] + self.mean
        recon_loss = np.linalg.norm(recon - embeddings, axis=1)
        logger.info(f"SVD recon loss: {recon_loss}")
        return Z
