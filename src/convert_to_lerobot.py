"""
Prepare trajectories for finetuning (convert to lerobot for pi)

Note: use this script to save tuning data directly to local. if data too big
for vram, use save_finetune_traj.py for saving relevant trajectories
to export to different machine.
"""

import os
import shutil

import pickle as pkl
import numpy as np
import tensorflow_datasets as tfds

from fire import Fire
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

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
         repo_name="hchen/libero"):
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

    # initialize lerobot dataset
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)
    
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float64",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float64",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # convert to lerobot format
    for iter_trajs in reverse_traj_dir:
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

            if env_traj_obj_repaired_param in solution_set:
                # add to lerobot dataset
                env_sol_traj_map = create_sol_traj_map(env_traj_obj)
            
                _, env_traj = next(iter(env_sol_traj_map.items()))
                for success_traj in env_traj:
                    episode_len = np.array(success_traj["images"]).shape[0]
                    # lerobot requires actions to be f64
                    success_traj_actions_f64 = np.float64(success_traj["actions"])

                    for step in range(episode_len):
                        dataset.add_frame(
                            {
                                "image": success_traj["images"][step],
                                "wrist_image": success_traj["wrist_images"][step],
                                "state": success_traj["states"][step],
                                "actions": success_traj_actions_f64[step],
                                "task": success_traj["prompt"]
                            }
                        )
                    
                    dataset.save_episode()

if __name__ == "__main__":
    Fire(main)

# python -m src.prep_traj --result_dir="./results/2025-12-24_03-06-14_generate_env_latent_2d_cma_me_task_5"