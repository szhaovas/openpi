from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ):
        super().__init__()

        # Encoder
        self._encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self._hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
        self._hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self._latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self._decoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self._outputs = nn.Linear(hidden_dim, input_dim)

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

    def _encode(
        self, x: Tensor, seq_len: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): Batch of inputs to be encoded; has shape (B, T, D).
            seq_len (Tensor): Optional sequence lengths of the input batch; has
                shape (B,). This is needed during training so that LSTM doesn't
                encode padded steps.
        """
        if seq_len is not None:
            # Ignore padded steps
            packed = pack_padded_sequence(
                x,
                seq_len.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self._encoder_lstm(packed)
        else:
            _, (h_n, _) = self._encoder_lstm(x)

        h_n = h_n[-1]
        mu = self._hidden_to_mu(h_n)
        logvar = self._hidden_to_logvar(h_n)
        return mu, logvar

    def _decode(self, z: Tensor, seq_len: int) -> Tensor:
        """
        Args:
            z (Tensor): Batch of latent embeddings to be decoded; has shape
                (B, :attr:`latent_dim`).
            seq_len (int): The number of steps to decode. This should equal the
                length of the longest sequence in the batch.
        """
        hidden = self._latent_to_hidden(z).unsqueeze(0)  # (1, batch, hidden)
        cell = torch.zeros_like(hidden)
        decoder_input = torch.zeros(
            (z.size(0), seq_len, self.input_dim), device=z.device, dtype=z.dtype
        )
        decoder_output, _ = self._decoder_lstm(decoder_input, (hidden, cell))
        output = self._outputs(decoder_output)
        return output

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reconstruct(
        self, x: Tensor, seq_len: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encodes + decodes; meant for training."""
        mu, logvar = self._encode(x, seq_len)
        z = self._reparameterize(mu, logvar)
        x_recon = self._decode(z, x.size(1))
        return z, mu, logvar, x_recon

    def forward(
        self, x: Tensor, seq_len: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Only encodes; meant for getting embedding measures."""
        B, _, _ = x.shape
        if (seq_len is None) and (B != 1):
            raise ValueError("Must provide seq_len when doing batch inference")

        mu, logvar = self._encode(x, seq_len)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar


def lstmvae_loss(
    x: Tensor, x_recon: Tensor, mu: Tensor, logvar: Tensor, seq_len: Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LSTM version of the VAE loss. Masks out reconstruction losses on padded
    steps.

    Args:
        x (Tensor): Ground truth; has shape (B, T, D).
        x_recon (Tensor): Reconstructed output; has shape (B, T, D).
        mu (Tensor): Latent mean; has shape (B, latent_dim).
        logvar (Tensor): Latent log-variance; has shape (B, latent_dim).
        seq_len (Tensor): Sequence lengths of the ground truth batch; has shape
            (B,). This is needed for masking out reconstruction losses on
            padded steps.
    """
    _, T, _ = x.shape

    recon_error = (x_recon - x) ** 2
    mask = torch.arange(T, device=x.device)[None, :] < seq_len[:, None]
    recon_error *= mask.unsqueeze(-1)
    recon_loss = recon_error.sum() / mask.sum()

    kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    kl_loss = kl_per_sample.mean()

    loss = recon_loss + kl_loss
    return loss, recon_loss, kl_loss
