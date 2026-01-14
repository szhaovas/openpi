import os
import torch
import torch.nn as nn
import torch.optim as optim

import glob
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm, trange

from src.encoder.data import Rollout_Dataset, Rollout, convert_to_rollout_dataset, extract_pkl_details
from src.encoder.lstm_encoder import LSTMVAE, vae_loss

import matplotlib.pyplot as plt

# TODO(Shihan): The following parameters are copied from the AURORA paper. 
#   See https://arxiv.org/pdf/2504.01915 Table 7
#   We can move them into some config yaml later

input_dim = 2048       # Pi0fast embedding size (after aggregating)
hidden_dim = 128       # Might want more since our data is higher-dimension (OG: 128)
latent_dim = 2        # How many measures
seq_len = 220
batch_size = 16        # og (128), we only have a max of 570 data samples so batch size is smaller
num_epochs = 200
learning_rate = 0.01

EMBEDDING_DS_PKL = "embedding_ds.pkl"

def extract_split_dataloaders(embedding_ds_pkl,
                            train_prop,
                            test_eval_prop,
                            batch_size,
                            seed):
    # extract_pkl_details(embedding_ds_pkl)
    balanced_ds = convert_to_rollout_dataset(embedding_ds_pkl)

    n_total = len(balanced_ds)
    n_train = int(train_prop * n_total)
    n_test = int(test_eval_prop * n_total)
    n_eval = n_total - n_train - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds, eval_ds = random_split(balanced_ds, [n_train, n_test, n_eval], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, eval_loader

def train_lstm_vae(input_dim,
                    hidden_dim,
                    latent_dim,
                    model_save_path,
                    train_prop=0.7,
                    test_eval_prop=0.15,
                    batch_size=16,
                    num_epochs=200,
                    learning_rate=0.01,
                    seed=147):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: make this more generalizable for other types of VAE (MLP)
    train_loader, test_loader, eval_loader = extract_split_dataloaders(EMBEDDING_DS_PKL,
                                                                       train_prop,
                                                                       test_eval_prop,
                                                                       batch_size,
                                                                       seed)

    lstm_vae = LSTMVAE(input_dim=input_dim,
                       hidden_dim=hidden_dim,
                       latent_dim=latent_dim).to(device)
    
    optimizer = optim.Adam(lstm_vae.parameters(), lr=learning_rate)

    history = {"train": [], "test": [], "eval": []}
    for epoch in trange(num_epochs):
        lstm_vae.train()

        epoch_total_loss, epoch_recon_loss, epoch_kl_loss = 0, 0, 0
        for batch in train_loader:
            features = batch["features"]
            optimizer.zero_grad()

            x_recon, mu, logvar, z = lstm_vae(features)
            loss, recon_loss, kl_loss = vae_loss(features, x_recon, mu, logvar)
            epoch_total_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

            loss.backward()

            optimizer.step()
        
        train_loss = epoch_total_loss/len(train_loader)
        train_recon_loss = epoch_recon_loss/len(train_loader)
        train_kl_loss = epoch_kl_loss/len(train_loader)
        history["train"].append((train_loss, train_recon_loss, train_kl_loss))

        def evaluate(loader):
            lstm_vae.eval()
            tot, rec_sum, kl_sum = 0, 0, 0
            with torch.no_grad():
                for batch in loader:
                    features = batch["features"]
                    x_recon, mu, logvar, _ = lstm_vae(features)
                    total, rec, kl = vae_loss(features, x_recon, mu, logvar)
                    tot += total.item()
                    rec_sum += rec.item()
                    kl_sum += kl.item()
            return tot/len(loader), tot/len(loader), tot/len(loader)

        test_loss, test_rec, test_kl = evaluate(test_loader)
        eval_loss, eval_rec, eval_kl = evaluate(eval_loader)
        history["test"].append((test_loss, test_rec, test_kl))
        history["eval"].append((eval_loss, eval_rec, eval_kl))
    
    epochs = range(1, num_epochs + 1)

    train_total = [t[0] for t in history["train"]]
    test_total  = [t[0] for t in history["test"]]
    eval_total  = [t[0] for t in history["eval"]]

    # plot results
    os.makedirs(model_save_path, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_total, label="Train", linewidth=2)
    plt.plot(epochs, test_total, label="Test", linestyle="--")
    plt.plot(epochs, eval_total, label="Eval", linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("VAE Training / Test / Eval Loss Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, "lstm_vae_losses.png"), dpi=300)
    plt.close()

    # save model
    save_model_path = os.path.join(model_save_path, "lstm_vae.pt")

    torch.save({
        "model_state_dict": lstm_vae.state_dict(),
        "config": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
        },
        "history": history,
    }, save_model_path)

    return history
    
if __name__ == "__main__":
    train_lstm_vae(input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   latent_dim=latent_dim,
                   model_save_path="./src/encoder/saved_models/")