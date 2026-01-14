"""
Evaluate finetuned checkpoint on environments from QD archive
"""

import os
import json
import pickle as pkl
import numpy as np

from fire import Fire
from tqdm import tqdm
from pathlib import Path
from functools import partial

from ribs.visualize import grid_archive_heatmap
from ribs.archives import GridArchive
from matplotlib import pyplot as plt

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from src.qd.qd_helpers import save_heatmap, evaluate_single
from src.task.task_5 import Task_5

def check_finetune_ds(finetune_pkls_dir):
    # SANITY CHECK TO ENSURE FINETUNE DS MATCHES
    all_traj_used_cnt = 0
    for root, dirs, files in os.walk(finetune_pkls_dir):
        for name in files:
            if name.endswith(".pkl"):
                file_path = os.path.join(root, name)

                with open(file_path, "rb") as f:
                    data = pkl.load(f)

                saved_traj = next(iter(data.values()))
                if len(saved_traj) == 0:
                    print(saved_traj)
                
                all_traj_used_cnt += len(saved_traj)

    print(all_traj_used_cnt)

def load_scheduler(scheduler_pkl_fn):  
    with open(scheduler_pkl_fn, "rb") as f:
        scheduler = pkl.load(f)
    return scheduler

def extract_solutions(archive):
    solutions = archive.data(["solution"])["solution"]
    return solutions

def base_qd_results(finetune_pkls_dir,
                    entropy_archive,
                    save_dir):
    e_archive_solutions = entropy_archive.data(["solution"])["solution"]
    e_archive_measures = entropy_archive.data(["measures"])["measures"]
    e_archive_objective = entropy_archive.data(["objective"])["objective"]

    sr_archive = GridArchive(
        solution_dim=entropy_archive.solution_dim,
        dims=entropy_archive.dims,
        ranges=[[entropy_archive.lower_bounds[0], entropy_archive.upper_bounds[0]],
                [entropy_archive.lower_bounds[1], entropy_archive.upper_bounds[1]]]
    )

    env_success_map = []
    
    for root, dirs, files in os.walk(finetune_pkls_dir):
        for name in files:
            if name.endswith(".pkl"):
                file_path = os.path.join(root, name)

                with open(file_path, "rb") as f:
                    data = pkl.load(f)

                solution, success_trajs = next(iter(data.items()))
                success_rate = len(success_trajs)/5

                matches = np.where(np.all(e_archive_solutions == solution, axis=1))[0]
                e_solutions_idx = matches[0]
                e_measure = e_archive_measures[e_solutions_idx]

                sr_archive.add_single(
                    solution=solution,
                    measures=e_measure,
                    objective=success_rate
                )
            
                env_success_map.append({
                    "solution": list(solution),
                    "measure": list(e_measure),
                    "success_rate": success_rate
                })

    with open(os.path.join(save_dir, "base_archive.pkl"), "wb") as f:
        pkl.dump(sr_archive, f)

    save_heatmap(sr_archive, os.path.join(save_dir, "base_success_rates.png"), vmin=0)
    return env_success_map

def collect_finetuned_results(archive,
                              experiment_cfg,
                              save_dir):
    # sequential with the assumption that finetuned model would be faster (less timesteps required)
    # TODO: make this customizable for other tasks (put it in the params and hydra)
    task_5_bddl = (
        Path(get_libero_path("bddl_files"))
            / "custom"
            / "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.bddl"
    )

    TASK_ENV = partial(
        OffScreenRenderEnv,
        bddl_file_name=task_5_bddl,
        camera_heights=256,
        camera_widths=256
    )

    all_solutions = archive.data(["solution"])["solution"]
    all_measures = archive.data(["measures"])["measures"]

    sr_archive = GridArchive(
        solution_dim=archive.solution_dim,
        dims=archive.dims,
        ranges=[[archive.lower_bounds[0], archive.upper_bounds[0]],
                [archive.lower_bounds[1], archive.upper_bounds[1]]]
    )

    env_success_map = []

    # for continuing the run
    all_vid_paths = []
    if os.path.isdir(save_dir):
        all_vid_paths = [
            name for name in os.listdir(save_dir)
            if os.path.isdir(os.path.join(save_dir, name))
        ]
    else:
        os.makedirs(save_dir, exist_ok=False)

    with open(os.path.join(save_dir, "log.txt"), "a") as log_f:
        for i, solution in tqdm(enumerate(all_solutions)):
            if f"{solution}_vids" in all_vid_paths:
                print(f"{i}th iteration solution -- {solution} already evaluated, results are in /{solution}_vids")
                continue
            else:
                print(f"Continuing to evaluate at iteration: {i}, solution: {solution}")
                vid_save_dir = os.path.join(save_dir, f"{solution}_vids")
                os.makedirs(vid_save_dir, exist_ok=True)

            (
                _,
                _,
                success_rate,
                _,
                _
            ) = evaluate_single(
                params=solution,
                host=experiment_cfg["host"],
                port=experiment_cfg["port"],
                ntrials=experiment_cfg["ntrials"],
                TASK_ENV=TASK_ENV,
                max_steps=experiment_cfg["max_steps"],
                num_steps_wait=experiment_cfg["num_steps_wait"],
                replan_steps=experiment_cfg["replan_steps"],
                seed=experiment_cfg["seed"],
                video_save_dir=vid_save_dir,
                video_save_cnt=1
            )

            sr_archive.add_single(
                solution=solution,
                measures=all_measures[i],
                objective=success_rate
            )

            save_heatmap(sr_archive, os.path.join(save_dir, "finetuned_success_rates.png"), vmin=0)
            with open(os.path.join(save_dir, f"finetuned_archive_iter{i}.pkl"), "wb") as f:
                pkl.dump(sr_archive, f)

            env_success_map.append({
                "solution": solution,
                "measure": all_measures[i],
                "success_rate": success_rate
            })

            print(f"Success rate: {success_rate}")

            log_f.write(f"{solution}, {all_measures[i]}, {success_rate}\n")
            log_f.flush()

    with open(os.path.join(save_dir, "finetuned_archive.pkl"), "wb") as f:
        pkl.dump(sr_archive, f)

    save_heatmap(sr_archive, os.path.join(save_dir, "finetuned_success_rates.png"), vmin=0)
        
    return env_success_map

def main(result_dir,
         checkpoint):
    # TODO: lookup the last scheduler pkl
    experiment_cfgs = {
        "5k": {
            "host": "0.0.0.0",
            "port": 8000,
            "ntrials": 5,
            "max_steps": 220,
            "num_steps_wait": 10,
            "replan_steps": 5,
            "seed": 42,
            "save_dir": "5k_iter_checkpoint"
        }
    }

    eval_dir = "qd_eval_results"
    os.makedirs(eval_dir, exist_ok=True)

    scheduler_pkl_fn = os.path.join(result_dir, "scheduler_00000099.pkl")
    finetune_pkls_dir = os.path.join(result_dir, "finetune_pkls")

    scheduler = load_scheduler(scheduler_pkl_fn)
    e_archive = scheduler.archive

    finetuned_env_success_map = collect_finetuned_results(e_archive,
                                                          experiment_cfgs[checkpoint],
                                                          f"{eval_dir}/{experiment_cfgs[checkpoint]['save_dir']}")

    with open(os.path.join(eval_dir, "finetuned_success_rates.json"), "w") as f:
        json.dump(finetuned_env_success_map, f, indent=4)

    print(f"Saved finetuned env results in: {eval_dir}")

    base_env_success_map = base_qd_results(finetune_pkls_dir,
                                           e_archive,
                                           eval_dir)
    
    with open(os.path.join(eval_dir, "base_success_rates.json"), "w") as f:
        json.dump(base_env_success_map, f, indent=4)

    print(f"Saved base env results in: {eval_dir}")

if __name__ == "__main__":
    Fire(main)

# python -m src.eval.qd_eval --result_dir="./results/2025-12-24_03-06-14_generate_env_latent_2d_cma_me_task_5" --checkpoint="5k"