import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import torch
import tqdm
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from src.dataset_utils import (
    TempDataset,
    Trajectory,
    embedding_collate,
    padded_embedding_collate,
)

from ._lstmvae import LSTMVAE, lstmvae_loss
from ._vae import VAE, vae_loss

logger = logging.getLogger(__name__)


@dataclass
class EncoderModelConfig:
    model_type: str
    input_dim: int
    hidden_dim: int
    latent_dim: int


@dataclass
class EncoderTrainingConfig:
    training_dataset_path: str
    batch_size: int
    num_steps: int
    learning_rate: float
    save_to: Optional[str]


MODEL_REGISTRY: Dict[str, Type[Union[VAE, LSTMVAE]]] = {
    "vae": VAE,
    "lstm_vae": LSTMVAE,
}


class EncoderManager:
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        model_cfg: Optional[EncoderModelConfig] = None,
        train_cfg: Optional[EncoderTrainingConfig] = None,
    ):
        """
        One of:
            - ckpt_path
            - (model_cfg + train_cfg)

        must be provided.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if ckpt_path is not None:
            self._init_encoder_from_checkpoint(Path(ckpt_path))
        elif model_cfg is not None and train_cfg is not None:
            self._init_encoder(model_cfg)
            self.model_cfg = model_cfg
            logger.warning(
                "Since no checkpoint is specified, an encoder will be trained from scratch"
            )
            self._train_encoder(train_cfg)
        else:
            raise ValueError(
                "Provide either ckpt_path OR (model_cfg + train_cfg)"
            )

        self.model.eval()

    def _init_encoder(self, model_cfg: EncoderModelConfig) -> None:
        model_type = model_cfg.model_type
        model_cls = MODEL_REGISTRY[model_type]
        self.model = model_cls(
            input_dim=model_cfg.input_dim,
            hidden_dim=model_cfg.hidden_dim,
            latent_dim=model_cfg.latent_dim,
        ).to(self.device)

    def _init_encoder_from_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        model_cfg = EncoderModelConfig(**checkpoint["model_cfg"])
        self._init_encoder(model_cfg)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model_cfg = model_cfg

    def _train_encoder(self, train_cfg: EncoderTrainingConfig) -> None:
        dataset = TempDataset(dataset_dir=Path(train_cfg.training_dataset_path))

        if isinstance(self.model, VAE):
            collate_fn = embedding_collate
        elif isinstance(self.model, LSTMVAE):
            collate_fn = padded_embedding_collate
        else:
            raise RuntimeError

        train_loader = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=train_cfg.learning_rate
        )

        self.model.train()

        for i in tqdm.trange(0, train_cfg.num_steps):
            total_loss = 0.0

            for x in train_loader:
                x = x.to(self.device)

                if isinstance(self.model, VAE):
                    _, mu, logvar, x_recon = self.model.reconstruct(x)
                    loss, _, _ = vae_loss(x, x_recon, mu, logvar)
                elif isinstance(self.model, LSTMVAE):
                    _, mu, logvar, x_recon = self.model.reconstruct(
                        x["padded_embeddings"], seq_len=x["pre_pad_lengths"]
                    )
                    loss, _, _ = lstmvae_loss(
                        x, x_recon, mu, logvar, seq_len=x["pre_pad_lengths"]
                    )
                else:
                    raise RuntimeError

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            tqdm.tqdm.write(f"Training step {i:05d} | loss={avg_loss:.4f}")

        if train_cfg.save_to is not None:
            self._save_checkpoint(Path(train_cfg.save_to))

    def _save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "model_cfg": self.model_cfg,
                "state_dict": self.model.state_dict(),
            },
            path,
        )
        logger.warning(f"Saved encoder checkpoint to {path}")

    @torch.no_grad()
    def encode(self, trajectories: List[Trajectory]) -> NDArray:
        if isinstance(self.model, VAE):
            x = embedding_collate(trajectories)
            z, _, _ = self.model(x)
        elif isinstance(self.model, LSTMVAE):
            x = padded_embedding_collate(trajectories)
            z, _, _ = self.model(
                x["padded_embeddings"], seq_len=x["pre_pad_lengths"]
            )
        else:
            raise RuntimeError

        return z.cpu().numpy()
