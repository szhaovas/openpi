import os
import pickle as pkl

import random
import torch
import torch.nn as nn
import torch.optim as optim

import glob
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def extract_pkl_details(embedding_ds_pkl):
    size_bytes = os.path.getsize(embedding_ds_pkl)
    size_mb = size_bytes / (1024 * 1024)

    with open(embedding_ds_pkl, "rb") as f:
        embedding_data = pkl.load(f)

    print(f"{'='*20} PICKLE SIZE {'='*20}")
    print(f"{size_mb} MB")
    print(f"{'='*20} DATA LENGTH {'='*20}")
    print(f"{len(embedding_data)} samples")
    print(f"{'='*20} DATA SAMPLE {'='*20}")
    print(embedding_data[0])
    print('='*50)

    return embedding_data

class Rollout:
    env_params: list
    env_id: int
    embeddings: torch.Tensor
    rollout_success: int

    def __post_init__(self):
        self.rollout_success = int(self.rollout_success)

class Rollout_Dataset(Dataset):
    def __init__(
        self,
        rollouts,
        device=torch.device("cuda")
    ):
        self.rollouts = rollouts
        self.length = len(rollouts)
        self.device = device

        features = pad_rollout_batch(self.rollouts, self.device)
        self.features = features

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = {
            'features': self.features[index]
        }
        return data
    
    def get_rollouts(self):
        return self.rollouts

    def get_features(self):
        return self.features
    
def pad_rollout_batch(
    rollouts, device = None
):
    """
    Pad the hidden states to the same length (max length in the batch).

    Args:
        rollouts: list of rollouts, each containing:
            - hidden_states (Tensor): shape [sequence_length, hidden_dim]
        device: device to put the tensors on

    Returns:
        padded_features (Tensor): shape [batch_size, max_length, hidden_dim]
    """
    # Extract all hidden states into a list
    batch_features = [r.embeddings for r in rollouts]

    # Determine padding dimensions
    max_length = max(seq.shape[0] for seq in batch_features)
    hidden_dim = batch_features[0].shape[-1]
    batch_size = len(batch_features)

    # Infer dtype and device from the first sequence
    dtype = batch_features[0].dtype
    if device is None:
        device = torch.device(batch_features[0].device)

    # Pre-allocate output tensors
    padded_features = torch.zeros(
        (batch_size, max_length, hidden_dim),
        dtype=dtype, device=device
    )

    # Fill in values for each sequence
    for i, seq in enumerate(batch_features):
        seq_length = seq.shape[0]
        padded_features[i, :seq_length] = seq.to(device)

    return padded_features

def balance_rollouts(rollouts, seed=174):
    rng = random.Random(seed)
    succ = [r for r in rollouts if int(r.rollout_success) == 1]
    fail = [r for r in rollouts if int(r.rollout_success) == 0]

    k = min(len(succ), len(fail))           
    succ_sel = rng.sample(succ, k)
    fail_sel = rng.sample(fail, k)

    balanced = succ_sel + fail_sel          
    return balanced

def convert_to_rollout_dataset(embedding_ds_pkl):
    with open(embedding_ds_pkl, "rb") as f:
        embedding_data = pkl.load(f)
        
    print("Converting embedding data to rollouts . . .")
    rollout_list = []
    for data_payload in embedding_data:
        rollout = Rollout()
        rollout.env_params = data_payload["env_params"]
        rollout.env_id = data_payload["env_id"]
        rollout.embeddings = torch.tensor(data_payload["embeddings"], dtype=torch.float32)
        rollout.rollout_success = data_payload["success"]

        rollout_list.append(rollout)
    
    balanced_rollouts = balance_rollouts(rollout_list)
    rollout_ds = Rollout_Dataset(balanced_rollouts)
    print(f"Final dataset size: {len(rollout_ds)}")

    return rollout_ds