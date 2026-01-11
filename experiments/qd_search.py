import csv
from functools import reduce
import logging
from operator import mul
import os
from pathlib import Path
import pickle as pkl
import re
from typing import Optional

from dask.distributed import Client
from dask.distributed import LocalCluster
import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap

from experiments.libero_spatial_eval import get_default_env_params
from experiments.myribs.schedulers import SchedulerExternal

logger = logging.getLogger(__name__)


def _extract_scheduler_itr(filename: str) -> Optional[int]:
    """Tries extracting the iteration number from a scheduler filename following
    the ``*/scheduler_[0-9]{8}.pkl`` format, where ``[0-9]{8}`` is the iteration
    number. If match fails, returns None.

    Args:
        filename (str): A scheduler filename following the
            ``*/scheduler_[0-9]{8}.pkl`` format.

    Returns:
        itr (int or None): Iteration number if match succeeds, else None.
    """
    pattern = r"scheduler_(\d{8})\.pkl"
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def save_heatmap(archive: GridArchive, heatmap_path: str):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=1, cmap="viridis")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("oc.env_int", lambda name: int(os.environ[name]))
    OmegaConf.register_new_resolver("oc.len", lambda x: len(x))
    OmegaConf.register_new_resolver("oc.mul", lambda *args: reduce(mul, args, 1))

    logdir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    summary_filename = logdir / "summary.csv"

    solution_dim = len(get_default_env_params())

    if cfg["reload_qd_scheduler_from"] is None:
        main_archive = instantiate(
            cfg["qd"]["archive"],
            solution_dim=solution_dim,
            extra_fields={"trajectories": ((cfg["eval"]["task_eval"]["num_trials_per_sol"],), object)},
        )
        passive_archive = instantiate(
            cfg["qd"]["archive"],
            solution_dim=solution_dim,
            learning_rate=None,
            threshold_min=-np.inf,
            extra_fields={"trajectories": ((cfg["eval"]["task_eval"]["num_trials_per_sol"],), object)},
        )

        emitters = [
            instantiate(
                cfg["qd"]["emitter"],
                archive=main_archive,
                x0=get_default_env_params(tid),
            )
            for tid in cfg["eval"]["task_ids"]
        ]

        # TODO: Maybe use bandit scheduler to prioritize some task_ids
        scheduler = SchedulerExternal(main_archive, emitters, result_archive=passive_archive)

        with open(summary_filename, "w") as summary_file:
            writer = csv.writer(summary_file)
            writer.writerow(["Iteration", "QD-Score", "Coverage", "Maximum", "Average"])
    else:
        reload_itr = _extract_scheduler_itr(cfg["reload_qd_scheduler_from"])
        if reload_itr is None:
            logger.error(
                "Received invalid reload_qd_scheduler_from parameter "
                f"{cfg['reload_qd_scheduler_from']}; "
                "expected */scheduler_[0-9]{8}.pkl"
            )
        with open(file=cfg["reload_qd_scheduler_from"], mode="rb") as f:
            scheduler = pkl.load(f)

    cluster = LocalCluster(
        processes=True,
        n_workers=cfg["eval"]["task_eval"]["num_trials_per_sol"],
        threads_per_worker=1,
    )
    client = Client(cluster)
    evaluators = [
        instantiate(cfg["eval"]["task_eval"], task_id=tid, dask_client=client) for tid in cfg["eval"]["task_ids"]
    ]

    start = 1 if cfg["reload_qd_scheduler_from"] is None else reload_itr + 1
    end = start + cfg["qd_search_itr"]
    for i in range(start, end):
        solutions = scheduler.ask()

        all_repaired, all_objective, all_measures, all_trajectories = [], [], [], []
        for eval_id in range(len(cfg["eval"]["task_ids"])):
            sol_start = eval_id * cfg["qd"]["emitter"]["batch_size"]
            sol_end = sol_start + cfg["qd"]["emitter"]["batch_size"]
            repaired, objective, measures, trajectories = evaluators[eval_id].evaluate(
                solutions=solutions[sol_start:sol_end]
            )

            all_repaired.extend(repaired)
            all_objective.extend(objective)
            all_measures.extend(measures)
            all_trajectories.extend(trajectories)

        all_repaired = np.array(all_repaired)
        all_objective = np.array(all_objective)
        all_measures = np.array(all_measures)

        edit_dists = np.linalg.norm(all_repaired - solutions, axis=1)
        # Solutions that have been modified by external repair are no longer
        # strictly emitted by CMA-ES and are injected into its update.
        injected = edit_dists > 0
        logger.info(f"{np.sum(injected)}/{injected.shape[0]} solutions have been modified by external repair")
        logger.info(f"\nedit_dists={edit_dists}")

        scheduler.tell(
            # Penalize objective with MILP editing distance if there is any
            np.clip(all_objective - edit_dists, a_min=0, a_max=None),
            all_measures,
            solution=all_repaired,
            injected=injected,
            trajectories=all_trajectories,
        )

        logger.info(
            f"------------------------ Iteration{i} ------------------------\n"
            f"\t QD-Score: {scheduler.result_archive.stats.qd_score}\n"
            f"\t Coverage: {scheduler.result_archive.stats.coverage}\n"
            f"\t Maximum : {scheduler.result_archive.stats.obj_max}\n"
            f"\t Average : {scheduler.result_archive.stats.obj_mean}\n"
        )

        final_itr = i == end
        if i % cfg["qd_search_log_every"] == 0 or final_itr:
            pkl.dump(
                scheduler,
                open(logdir / f"scheduler_{i:08d}.pkl", "wb"),
            )

            with open(summary_filename, "a") as summary_file:
                writer = csv.writer(summary_file)
                data = [
                    i,
                    scheduler.result_archive.stats.qd_score,
                    scheduler.result_archive.stats.coverage,
                    scheduler.result_archive.stats.obj_max,
                    scheduler.result_archive.stats.obj_mean,
                ]
                writer.writerow(data)

            save_heatmap(
                scheduler.result_archive,
                logdir / f"heatmap_{i:08d}.png",
            )


if __name__ == "__main__":
    main()
