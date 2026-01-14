import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import glob
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# --- LSTM VAE Autoencoder ---
class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMVAE, self).__init__()

        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.outputs = nn.Linear(hidden_dim, input_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_n = h_n[-1] 
        mu = self.hidden_to_mu(h_n)
        logvar = self.hidden_to_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        hidden = self.latent_to_hidden(z).unsqueeze(0)  # (1, batch, hidden)
        cell = torch.zeros_like(hidden)
        decoder_input = torch.zeros((z.size(0), seq_len, self.input_dim)).to(z.device) 
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        output = self.outputs(decoder_output)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, x.size(1))
        return x_recon, mu, logvar, z

# --- Loss function ---
def vae_loss(x, x_recon, mu, logvar):
    # Recon: mean over all elements (B*T*D)
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    # KL: mean over batch
    kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)  # (B,)
    kl_loss = kl_per_sample.mean()
    loss = recon_loss + kl_loss
    return loss, recon_loss, kl_loss
