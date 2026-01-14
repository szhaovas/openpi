"""
Save trajectories used for finetuning to pkl (easier to transfer between machines)
"""

import os
import shutil

import pickle as pkl
import numpy as np
from tqdm import tqdm
# import tensorflow_datasets as tfds

from fire import Fire

def create_sol_traj_map(env_traj_obj):
    """
    input data:
    env_traj_obj: {
        repaired_params: list,
        original_parmas: list, # typo in data collection
        edit_dist: int,
        success_rate: float,
        trials: [
            {
                prompt: str,
                images: list,
                wrist_images: list,
                states: list,
                actions: list,
                success: bool,
                embeddings: list
            }
        ]
    }

    return data:
    sol_traj_map: {
        repaired_params: [
            {
                prompt,
                images,
                wrist_images,
                states,
                actions
            }
        ]
    }
    """
    sol_traj_map = {}
    env_param_key = tuple(env_traj_obj["repaired_params"])
    sol_traj_map[env_param_key] = []
    for trial_id, trial in enumerate(env_traj_obj["trials"]):
        if trial["success"]:
            sol_traj_map[env_param_key].append({
                "prompt": trial["prompt"],
                "images": trial["images"],
                "wrist_images": trial["wrist_images"],
                "states": trial["states"],
                "actions": trial["actions"]
            })
    
    return sol_traj_map

def main(result_dir,
         save_dir="finetune_pkls"):
    # load in original dataset
    traj_dir = os.path.join(result_dir, "trajectories")
    # TODO: lookup the last scheduler pkl
    scheduler_pkl = os.path.join(result_dir, "scheduler_00000099.pkl")

    with open(scheduler_pkl, "rb") as f:
        scheduler = pkl.load(f)
        archive = scheduler.archive
        archive_solutions = archive.data(["solution"])["solution"]

    solution_set = set()
    for solution in archive_solutions:
        solution_set.add(tuple(solution))

    reverse_traj_dir = sorted(
        [
            d for d in os.listdir(traj_dir)
            if d.startswith("iter_") and os.path.isdir(os.path.join(traj_dir, d))
        ],
        reverse=True
    )

    # fill finetune pkl
    save_dir = os.path.join(result_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving finetuning trajectories to {save_dir}")

    seen_params = set()

    count = 0
    for iter_trajs in tqdm(reverse_traj_dir):
        iter_trajs_path = os.path.join(traj_dir, iter_trajs)
        trajs_pkls = sorted(
            f for f in os.listdir(iter_trajs_path)
            if f.endswith(".pkl")
        )

        for traj_pkl in trajs_pkls:
            traj_pkl_path = os.path.join(iter_trajs_path, traj_pkl)

            with open(traj_pkl_path, "rb") as f:
                env_traj_obj = pkl.load(f)
            
            env_traj_obj_repaired_param = tuple(env_traj_obj["repaired_params"])

            if env_traj_obj_repaired_param in solution_set and env_traj_obj_repaired_param not in seen_params:
                seen_params.add(env_traj_obj_repaired_param)

                env_sol_traj_map = create_sol_traj_map(env_traj_obj)
                params, env_traj = next(iter(env_sol_traj_map.items()))

                with open(os.path.join(save_dir, f"{params}.pkl"), "wb") as f:
                    pkl.dump(env_sol_traj_map, f)
                
                count += 1
                print(f"Saved {count} trajectories!")

if __name__ == "__main__":
    Fire(main)

# python -m src.save_finetune_traj --result_dir="./results/2025-12-24_03-06-14_generate_env_latent_2d_cma_me_task_5"