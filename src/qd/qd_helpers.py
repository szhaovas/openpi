import os
import collections
import torch
import math
import imageio

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from ribs.visualize import grid_archive_heatmap
from tqdm import tqdm, trange
from dask.distributed import Client, LocalCluster, get_worker

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from src.encoder.lstm_encoder import LSTMVAE, vae_loss
from src.task.libero_helpers import _quat2axisangle

def save_heatmap(archive, heatmap_path, vmin=-1):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=vmin, vmax=1, cmap="viridis")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())

def collect_pi0fast_embedding(action_data):
    # using best ablation settings from SAFE paper
    pre_logits = action_data["pre_logits"]
    return np.mean(pre_logits, axis=0)

def evaluate_single(params,
                    host,
                    port,
                    ntrials,
                    TASK_ENV,
                    max_steps,
                    num_steps_wait,
                    replan_steps,
                    seed,
                    save_traj_metadata=None,
                    collect_embeddings=False,
                    video_save_dir=None,
                    video_save_cnt=1):
    # start the job
    try:
        worker = get_worker()
        print(
            f"[START] Worker: {worker.address} | pid={os.getpid()} running trajectory \
                for params, {params}, with seed {seed}",
            flush=True
        )
    except:
        print(
            f"[START] Single non-parallelized for {params} with seed {seed}"
        )

    np.random.seed(seed)

    # load openpi
    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    # load environment
    original_params = params
    env = TASK_ENV(
        env_params=params,
        # repair_env=True,
        repair_config={
            'time_limit':1500,
            'seed':seed
        }
    )
    env.seed(seed)

    # things to keep track of
    all_trial_embeddings = [] # for calculating measures
    success_rate = 0.0

    all_traj = {
        "repaired_params": None,
        "original_parmas": original_params,
        "edit_dist": 0.0,
        "success_rate": 0.0,
        "trials": [],
    } 

    for trial in range(ntrials):
        action_plan = collections.deque()
        
        obs, repaired_params = env.reset()

        # things to store
        edit_dist = np.linalg.norm(original_params - repaired_params)
        all_traj["repaired_params"] = repaired_params
        all_traj["edit_dist"] = edit_dist
        trial_i_traj = {
            "prompt": None,
            "images": [],
            "wrist_images": [],
            "states": [],
            "actions": [],
            "success": False,
            "embeddings": []
        }
        trial_i_embeddings = []

        replay_images = []
        replay_wrist_images = []

        for t in range(max_steps + num_steps_wait):
            try:
                if t < num_steps_wait:
                    obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                replay_images.append(img)
                replay_wrist_images.append(wrist_img)
            
                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": env.language_instruction,
                        "run/env_id": -1
                    }

                    action_data = client.infer(element)

                    embedding_t = 0
                    if collect_embeddings:
                        embedding_t = collect_pi0fast_embedding(action_data)

                    action_chunk = np.squeeze(action_data["actions"])
                    
                    assert (
                        len(action_chunk) >= replan_steps
                    ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    
                    action_plan.extend(action_chunk[: replan_steps])
                
                action = action_plan.popleft()

                # storing trajectory
                trial_i_traj["prompt"] = env.language_instruction
                trial_i_traj["images"].append(img)
                trial_i_traj["wrist_images"].append(wrist_img)
                trial_i_traj["states"].append(np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                ))
                trial_i_traj["actions"].append(action)
                trial_i_traj["embeddings"].append(embedding_t)

                # store embeddings for later
                trial_i_embeddings.append(embedding_t)

                # prep for next step
                obs, reward, done, info = env.step(action.tolist())
                
                if done:
                    success_rate += 1
                    trial_i_traj["success"] = True
                    break
            except Exception as e:
                print(e)
                break
        
        all_trial_embeddings.append(trial_i_embeddings)
        all_traj["trials"].append(trial_i_traj)

        if video_save_dir and video_save_cnt > 0:
            img_save_path = os.path.join(video_save_dir, f"table_trial_{trial}_success_{trial_i_traj['success']}.mp4")
            imageio.mimwrite(
                img_save_path,
                [np.asarray(x) for x in replay_images],
                fps=30
            )
            wrist_img_save_path = os.path.join(video_save_dir, f"wrist_trial_{trial}_success_{trial_i_traj['success']}.mp4")
            imageio.mimwrite(
                wrist_img_save_path,
                [np.asarray(x) for x in replay_wrist_images],
                fps=30
            )
            video_save_cnt -= 1

    client._ws.close()

    success_rate /= ntrials
    success_rate = np.clip(success_rate, 1e-6, 1 - 1e-6)
    all_traj["success_rate"] = success_rate

    entropy = -success_rate*math.log2(success_rate) - (1-success_rate)*math.log2(1-success_rate)
    unpadded_torch_embeddings = [torch.as_tensor(e) for e in all_trial_embeddings]

    if save_traj_metadata:
        # assumption is that later trajectories are the ones to keep (in data clean step)
        # storing here to prevent overhead from returning big thing in dask
        iter_i_save_dir = os.path.join(save_traj_metadata["traj_save_dir"], f"iter_{save_traj_metadata['iteration']:08d}")
        os.makedirs(iter_i_save_dir, exist_ok=True)
        with open(os.path.join(iter_i_save_dir, f"trajectories_seed_{seed}.pkl"), "wb") as f:
            pkl.dump(all_traj, f)

    try:
        print(
            f"[END] Worker: {worker.address} | pid={os.getpid()} running trajectory \
                for params, {params}, with seed {seed}",
            flush=True
        )
    except:
        print(
            f"[END] Single non-parallelized for {params} with seed {seed}"
        )

    return (
        repaired_params,
        # for obj
        edit_dist, 
        success_rate, 
        entropy,
        # for measures
        unpadded_torch_embeddings,
    )
    