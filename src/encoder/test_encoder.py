import numpy as np
import torch
from src.encoder.lstm_encoder import LSTMVAE, vae_loss
from src.encoder.train import extract_split_dataloaders

def load_lstm(lstm_save_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(lstm_save_file, map_location=device)

    config = checkpoint["config"]
    model = LSTMVAE(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def latent_ranges():
    train_loader, _, _ = extract_split_dataloaders("embedding_ds.pkl",
                                                   0.9,
                                                   0.05,
                                                   batch_size=1,
                                                   seed=42)

    lstm_vae_path = "./src/encoder/saved_models/lstm_vae_10d.pt"
    lstm_vae = load_lstm(lstm_vae_path)

    all_compressed_embeddings = []
    for batch in train_loader:
        features = batch["features"]
        x_recon, mu, logvar, z = lstm_vae(features)
        z_np = z.detach().cpu().numpy()
        all_compressed_embeddings.append(z_np)

    all_compressed_embeddings = np.array(all_compressed_embeddings)
    all_compressed_embeddings = np.squeeze(all_compressed_embeddings)
    feature_mins = all_compressed_embeddings.min(axis=0)
    feature_maxs = all_compressed_embeddings.max(axis=0)
    ranges = np.stack([feature_mins, feature_maxs], axis=1)

    return ranges
