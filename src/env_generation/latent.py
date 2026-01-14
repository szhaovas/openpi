import os
import collections
import torch
import logging
import pickle as pkl

import numpy as np

from tqdm import tqdm, trange
from dask.distributed import Client, LocalCluster, get_worker

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from src.encoder.lstm_encoder import LSTMVAE, vae_loss
from src.qd.qd_helpers import evaluate_single

class Latent_Evaluator():
    def __init__(self, name, config):
        self.name = name

        # loading eval params
        self.host = config.host
        self.port = config.port
        self.num_workers = config.num_workers
        self.ntrials = config.ntrials
        self.seed = config.seed
        self.num_steps_wait = config.num_steps_wait
        self.replan_steps = config.replan_steps

        # loading lstm encoder
        self.max_traj_len = config.max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(config.lstm_vae_fn, map_location=self.device, weights_only=False)

        model_config = checkpoint["config"]
        self.model = LSTMVAE(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            latent_dim=model_config["latent_dim"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        # loading dask
        # mute the logs
        logging.getLogger("distributed").setLevel(logging.ERROR)
        logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)
        logging.getLogger("distributed.core").setLevel(logging.ERROR)
        logging.getLogger("dask").setLevel(logging.ERROR)

        self.cluster = LocalCluster(
            processes=True,
            n_workers=self.num_workers,
            threads_per_worker=1
        )
        self.dask_client = Client(self.cluster)

        print(f"Loaded {self.name} evaluation module")

    def pad_single_embedding(self, embedding):
        T, D = embedding.shape
        dtype = embedding.dtype

        padded = torch.zeros((self.max_traj_len, D), dtype=dtype, device=self.device)
        padded[:T] = embedding.to(self.device)
        return padded

    def pad_batch_embeddings(self, embeddings):
        D = embeddings[0].shape[-1]
        dtype = embeddings[0].dtype
        B = len(embeddings)

        padded = torch.zeros((B, self.max_traj_len, D), dtype=dtype, device=self.device)

        for i, emb in enumerate(embeddings):
            T = min(emb.shape[0], self.max_traj_len)  
            padded[i, :T] = emb[:T].to(self.device)

        return padded
    
    # extract measures
    def compress_embeddings(self, padded_embeddings):
        x_recon, mu, logvar, compressed_embedding = self.model(padded_embeddings)
        loss, recon_loss, kl_loss = vae_loss(padded_embeddings, x_recon, mu, logvar)

        # average out the compressed embeddings
        avg_compressed_embedding = compressed_embedding.mean(dim=0) 

        return (
            avg_compressed_embedding.detach().cpu().numpy(),
            loss.detach().cpu().numpy(),
            recon_loss.detach().cpu().numpy(),
            kl_loss.detach().cpu().numpy()
        )
    
    # evaluation parallelized
    def evaluate(self,
                 all_params,
                 TASK_ENV,
                 save_traj_metadata):
        all_repaired_params, all_edit_dist, all_success_rate, all_entropy, all_embeddings = [], [], [], [], []
        futures = [
            self.dask_client.submit(
                evaluate_single,
                params=params,
                host=self.host,
                port=self.port,
                ntrials=self.ntrials,
                TASK_ENV=TASK_ENV,
                max_steps=self.max_traj_len,
                num_steps_wait=self.num_steps_wait,
                replan_steps=self.replan_steps,
                seed=self.seed+i,
                save_traj_metadata=save_traj_metadata
            )
            for i, params in enumerate(all_params)
        ]
        results = self.dask_client.gather(futures)

        for (
            repaired_params,
            edit_dist,
            success_rate,
            entropy,
            unpadded_embeds
        ) in results:
            all_repaired_params.append(repaired_params)
            all_edit_dist.append(edit_dist)
            all_success_rate.append(success_rate)
            all_entropy.append(entropy)
            all_embeddings.append(unpadded_embeds)

        # compress embeddings (measure calculation)
        padded_embeddings = [self.pad_batch_embeddings(trial_embeddings) for trial_embeddings in all_embeddings]
        # padded_embeddings_np = [
        #     pe.detach().cpu().numpy() if torch.is_tensor(pe) else pe
        #     for pe in padded_embeddings
        # ]
    
        all_compressed_embeds, all_vae_loss = [], []
        for embed in padded_embeddings:
            compressed_embed, total_loss, recon_loss, kl_loss = self.compress_embeddings(embed)
            
            all_compressed_embeds.append(compressed_embed)
            all_vae_loss.append((total_loss, recon_loss, kl_loss))

        # objective (entropy - edit dist)
        all_entropy = np.array(all_entropy)
        all_edit_dist = np.array(all_edit_dist)

        all_obj = all_entropy - all_edit_dist

        return (
            # for storing in archive
            np.array(all_repaired_params),
            np.array(all_compressed_embeds),
            np.array(all_obj),
            # for logging
            np.array(all_vae_loss),
            np.array(all_success_rate)
        )