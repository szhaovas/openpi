from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ):
        super().__init__()

        # Encoder
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self._hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
        self._hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def _encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): Batch of inputs to be encoded; has shape (B, D).
        """
        h = self._encoder(x)
        mu = self._hidden_to_mu(h)
        logvar = self._hidden_to_logvar(h)
        return mu, logvar

    def _decode(self, z: Tensor) -> Tensor:
        """
        Args:
            z (Tensor): Batch of latent embeddings to be decoded; has shape
                (B, :attr:`latent_dim`).
        """
        return self._decoder(z)

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reconstruct(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encodes + decodes; meant for training."""
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        x_recon = self._decode(z)
        return z, mu, logvar, x_recon

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Only encodes; meant for getting embedding measures."""
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar


def vae_loss(
    x: Tensor, x_recon: Tensor, mu: Tensor, logvar: Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        x (Tensor): Ground truth; has shape (B, D).
        x_recon (Tensor): Reconstructed output; has shape (B, D).
        mu (Tensor): Latent mean; has shape (B, latent_dim).
        logvar (Tensor): Latent log-variance; has shape (B, latent_dim).
    """
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl.sum(dim=1).mean()

    loss = recon_loss + kl_loss
    return loss, recon_loss, kl_loss
