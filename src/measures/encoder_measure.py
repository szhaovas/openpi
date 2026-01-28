import logging
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

from .encoders._lstmvae import LSTMVAE, lstmvae_loss
from .encoders._vae import VAE, vae_loss
from .measure_model import MeasureModel, ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


MODEL_REGISTRY: Dict[str, Type[Union[VAE, LSTMVAE]]] = {
    "vae": VAE,
    "lstm_vae": LSTMVAE,
}


class EncoderMeasure(MeasureModel):
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        model_cfg: Optional[ModelConfig] = None,
        train_cfg: Optional[TrainingConfig] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(ckpt_path, model_cfg, train_cfg)

        self.model.eval()

    def _init(self, model_cfg: ModelConfig) -> None:
        model_type = model_cfg.model_type
        if model_type not in ["vae", "lstm_vae"]:
            raise ValueError(
                f"Unknown model_type {model_type} (must be one of {['vae', 'lstm_vae']})"
            )
        model_cls = MODEL_REGISTRY[model_type]
        self.model = model_cls(
            input_dim=model_cfg.input_dim,
            hidden_dim=model_cfg.hidden_dim,
            latent_dim=model_cfg.latent_dim,
        ).to(self.device)

    def _init_from_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.model_cfg = checkpoint["model_cfg"]
        self._init(self.model_cfg)
        self.model.load_state_dict(checkpoint["state_dict"])

    def _train(self, train_cfg: TrainingConfig) -> None:
        dataset = TempDataset(dataset_dir=Path(train_cfg.training_dataset_path))

        if isinstance(self.model, VAE):
            collate_fn = embedding_collate
        elif isinstance(self.model, LSTMVAE):
            collate_fn = padded_embedding_collate
        else:
            raise RuntimeError

        data_loader = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=train_cfg.learning_rate
        )

        self.model.train()

        for i in tqdm.trange(0, train_cfg.num_train_epochs):
            total_loss = 0.0

            for x in data_loader:
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

            avg_loss = total_loss / len(data_loader)
            tqdm.tqdm.write(f"Epoch {i:04d} | loss={avg_loss:.4f}")

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
    def compute_measures(self, trajectories: List[Trajectory]) -> NDArray:
        if isinstance(self.model, VAE):
            x = embedding_collate(trajectories).to(self.device)
            z, _, _ = self.model(x)
        elif isinstance(self.model, LSTMVAE):
            x = padded_embedding_collate(trajectories)
            z, _, _ = self.model(
                x["padded_embeddings"].to(self.device),
                seq_len=x["pre_pad_lengths"].to(self.device),
            )
        else:
            raise RuntimeError

        return z.cpu().numpy()
