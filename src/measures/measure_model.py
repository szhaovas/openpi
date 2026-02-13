import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from omegaconf import DictConfig

from src.dataset_utils import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_type: str
    input_dim: int
    latent_dim: int
    _extras: Optional[DictConfig]


@dataclass
class TrainingConfig:
    training_dataset_path: str
    save_to: Optional[str]
    _extras: Optional[DictConfig]


class MeasureModel(ABC):
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        model_cfg: Optional[ModelConfig] = None,
        train_cfg: Optional[TrainingConfig] = None,
    ):
        """Handles initialization and training of measure models.

        Args:
            One of:
                - ckpt_path
                    - A measure model is directly loaded from the checkpoint
                - (model_cfg + train_cfg)
                    - Initializes and trains a fresh measure model
        """
        if ckpt_path is not None:
            self._init_from_checkpoint(Path(ckpt_path))
        elif model_cfg is not None and train_cfg is not None:
            self._init(model_cfg)
            self.model_cfg = model_cfg
            logger.warning(
                "Since no checkpoint is specified, a measure model will be trained from scratch"
            )
            self._train(train_cfg)
        else:
            raise ValueError(
                "Provide either ckpt_path OR (model_cfg + train_cfg)"
            )

    @abstractmethod
    def _init(self, model_cfg: ModelConfig) -> None:
        """ """

    @abstractmethod
    def _init_from_checkpoint(self, path: Path) -> None:
        """ """

    @abstractmethod
    def _train(self, train_cfg: TrainingConfig) -> None:
        """ """

    @abstractmethod
    def _save_checkpoint(self, path: Path) -> None:
        """ """

    @abstractmethod
    def compute_measures(self, trajectories: List[Trajectory]) -> np.ndarray:
        """ """
