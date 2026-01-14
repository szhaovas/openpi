import os
import hydra
import json
import sys
import logging
import csv
import time
import shutil
import numpy as np
import pandas as pd
import pickle as pkl
import copy

from omegaconf import DictConfig, OmegaConf
from tqdm import trange, tqdm
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from src.qd.qd_helpers import save_heatmap

# python -m src.generate_env env_search=latent_2d qd=cma_me task=task_5
@hydra.main(version_base=None, config_path="../config", config_name="env_gen_config")
def main(cfg: DictConfig):
    # load log/save configs
    logger = logging.getLogger(__name__)
    run_dir = HydraConfig.get().run.dir

    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    summary_csv_fn = os.path.join(run_dir, cfg.summary_csv)
    trajectory_save_dir = os.path.join(run_dir, cfg.trajectory_save_dir)

    os.makedirs(trajectory_save_dir, exist_ok=True)

    # load LIBERO env
    libero_env_obj = instantiate(cfg.task)
    TASK_ENV = libero_env_obj.get_task_env()

    # load QD components
    qd_algo_obj = instantiate(cfg.qd)
    scheduler = qd_algo_obj.get_scheduler()

    # load env search (measure/obj) components
    evaluation_obj = instantiate(cfg.env_search)

    # QD loop
    num_iters = cfg.num_iters
    with open(summary_csv_fn, "a") as summary_csv:
        csv_writer = csv.writer(summary_csv)
        csv_writer.writerow(["Iteration", "QD-Score", "Coverage", "Maximum", "Average", "Average-VAE-loss", "Average-success-rate"])

        for i in trange(num_iters):
            solutions = scheduler.ask()

            (
                # for archive
                repaired_solutions,
                compressed_embeds,
                objectives,
                # for logging
                vae_loss,
                success_rate,
            ) = evaluation_obj.evaluate(solutions,
                                        TASK_ENV,
                                        {
                                            "traj_save_dir": trajectory_save_dir,
                                            "iteration": i
                                        })
            
            scheduler._cur_solutions = repaired_solutions
            scheduler.tell(objectives, compressed_embeds)

            # logging
            if i % log_freq == 0:
                logger.info(
                    f"\n------------------ Iteration{i} ------------------\n"
                    f"\t QD-Score: {scheduler.result_archive.stats.qd_score}\n"
                    f"\t Coverage: {scheduler.result_archive.stats.coverage}\n"
                    f"\t Maximum : {scheduler.result_archive.stats.obj_max}\n"
                    f"\t Average : {scheduler.result_archive.stats.obj_mean}\n"
                    f"\t Average VAE Loss: {np.mean(vae_loss)}\n"
                    f"\t Average success rate: {np.mean(success_rate)}"                   
                )
                
                summary_data = [
                    i,
                    scheduler.result_archive.stats.qd_score,
                    scheduler.result_archive.stats.coverage,
                    scheduler.result_archive.stats.obj_max,
                    scheduler.result_archive.stats.obj_mean,
                    np.mean(vae_loss),
                    np.mean(success_rate)
                ]
                csv_writer.writerow(summary_data)
                summary_csv.flush()

                save_heatmap(
                    scheduler.archive,
                    os.path.join(f"{run_dir}/heatmap_{i:08d}.png")
                )

            # saving
            if i % save_freq == 0 or i == num_iters - 1:                
                # assumption is that later trajectories are the ones to keep (in data clean step)
                with open(os.path.join(run_dir, f"scheduler_{i:08d}.pkl"), "wb") as f:
                    pkl.dump(scheduler, f)

if __name__ == "__main__":
    main()