import csv
import logging
import os
import pickle as pkl
import re
from functools import reduce
from operator import add, mul
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from dask.distributed import Client, LocalCluster
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap

from src.dataset_utils import TempDataset
from src.libero_spatial_eval import get_default_env_params

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


def add_omegaconf_resolvers():
    OmegaConf.register_new_resolver(
        "oc.env_int", lambda name: int(os.environ[name])
    )
    OmegaConf.register_new_resolver(
        "oc.env_isone", lambda name: int(os.environ[name]) == 1
    )
    OmegaConf.register_new_resolver("oc.len", lambda x: len(x))
    OmegaConf.register_new_resolver(
        "oc.add", lambda *args: reduce(add, args, 1)
    )
    OmegaConf.register_new_resolver(
        "oc.mul", lambda *args: reduce(mul, args, 1)
    )


def seed_everything(seed: int):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_embeddings(colemb_cfg: DictConfig):
    embedding_dataset = TempDataset(dataset_dir=colemb_cfg.save_embeddings_to)

    archive = instantiate(
        colemb_cfg.envgen.archive, solution_dim=len(get_default_env_params())
    )

    emitters = [
        instantiate(
            colemb_cfg.envgen.emitter,
            archive=archive,
            x0=get_default_env_params(tid),
        )
        for tid in colemb_cfg.eval.task_ids
    ]
    for em, tid in zip(emitters, colemb_cfg.eval.task_ids):
        em.task_id = tid

    scheduler = instantiate(
        colemb_cfg.envgen.scheduler, archive=archive, emitters=emitters
    )

    if colemb_cfg.single_process:
        client = None
    else:
        cluster = LocalCluster(
            processes=True,
            n_workers=colemb_cfg.envgen.emitter.batch_size,
            threads_per_worker=1,
        )
        client = Client(cluster)

    evaluators = [
        instantiate(colemb_cfg.eval.task_eval, task_id=tid, dask_client=client)
        for tid in range(10)
    ]

    for _ in tqdm.trange(
        0,
        colemb_cfg.collect_embedding_num_evals,
        len(scheduler.emitters) * colemb_cfg.envgen.emitter.batch_size,
    ):
        solutions = scheduler.ask()

        for eid, em in enumerate(scheduler.emitters):
            sol_start = eid * colemb_cfg.envgen.emitter.batch_size
            sol_end = sol_start + colemb_cfg.envgen.emitter.batch_size
            trajectories = evaluators[em.task_id].get_single_trajectories(
                solutions=solutions[sol_start:sol_end]
            )

            for traj in trajectories:
                embedding_dataset.write_episode(trajectory=traj)

        # Dummy objective and measure values
        scheduler.tell(
            np.repeat(1e-6, solutions.shape[0]),
            np.random.rand(solutions.shape[0], 1),
            solution=solutions,
            injected=np.full(solutions.shape[0], False),
        )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    add_omegaconf_resolvers()
    seed_everything(cfg.seed)

    logdir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    summary_filename = logdir / "summary.csv"

    if cfg.reload_scheduler_from is None:
        archive = instantiate(
            cfg.envgen.archive,
            solution_dim=len(get_default_env_params()),
            extra_fields={
                "task_id": (
                    (),
                    np.int32,
                ),  # store task_id for each env for later eval
            },
        )

        # Each task_id gets a single emitter whose x0 is set to its
        # corresponding default.
        emitters = [
            instantiate(
                cfg.envgen.emitter,
                archive=archive,
                x0=get_default_env_params(tid),
            )
            for tid in cfg.eval.task_ids
        ]
        for em, tid in zip(emitters, cfg.eval.task_ids):
            em.task_id = tid

        scheduler = instantiate(
            cfg.envgen.scheduler, archive=archive, emitters=emitters
        )

        with open(summary_filename, "w") as summary_file:
            writer = csv.writer(summary_file)
            writer.writerow(
                ["Iteration", "QD-Score", "Coverage", "Maximum", "Average"]
            )
    else:
        reload_itr = _extract_scheduler_itr(cfg.reload_scheduler_from)
        if reload_itr is None:
            logger.error(
                "Received invalid reload_scheduler_from parameter "
                f"{cfg.reload_scheduler_from}; "
                "expected */scheduler_[0-9]{8}.pkl"
            )
            raise ValueError
        with open(file=cfg.reload_scheduler_from, mode="rb") as f:
            scheduler = pkl.load(f)

    succ_dataset_dir = (
        logdir / "succ_dataset"
        if cfg.write_to_succ_dataset is None
        else Path(cfg.write_to_succ_dataset)
    )
    temp_succ_dataset = TempDataset(dataset_dir=succ_dataset_dir)
    fail_dataset_dir = (
        logdir / "fail_dataset"
        if cfg.write_to_fail_dataset is None
        else Path(cfg.write_to_fail_dataset)
    )
    temp_fail_dataset = TempDataset(dataset_dir=fail_dataset_dir)

    if cfg.single_process:
        client = None
    else:
        cluster = LocalCluster(
            processes=True,
            n_workers=cfg.eval.task_eval.num_trials_per_sol,
            threads_per_worker=1,
        )
        client = Client(cluster)

    eval_mode = hydra.core.hydra_config.HydraConfig.get().runtime.choices[
        "eval"
    ]
    if eval_mode == "policy_embedding":
        collect_embedding_config = hydra.compose(
            config_name="main",
            overrides=["envgen=domain_randomization", "eval=first_timestep"],
        )
        collect_embeddings(collect_embedding_config)
        encoder_manager = instantiate(cfg.eval.measure_encoder)
    else:
        encoder_manager = None

    # Each task gets an evaluator. There are 10 libero-spatial tasks
    # TODO: Define an evaluator retrieval
    evaluators = [
        instantiate(
            cfg.eval.task_eval,
            task_id=tid,
            dask_client=client,
            measure_encoder=encoder_manager,
        )
        for tid in range(10)
    ]

    search_itr = cfg.env_search_num_evals / (
        len(cfg.eval.task_ids)
        * cfg.envgen.emitter.batch_size
        * cfg.eval.task_eval.num_trials_per_sol
    )
    assert search_itr > 1

    logger.info(
        f"Search evaluation budget {cfg.env_search_num_evals} needs to be split among "
        f"{len(cfg.eval.task_ids)} emitters.\n Each emitter samples batch_size "
        f"{cfg.envgen.emitter.batch_size} solutions,\n and each solution will "
        f"be evaluated {cfg.eval.task_eval.num_trials_per_sol} times.\n"
        "====================================================================\n"
        f"Up to {search_itr} search iterations will be run.\n"
        "====================================================================\n"
    )

    start = 1 if cfg.reload_scheduler_from is None else reload_itr + 1
    for i in tqdm.trange(start, search_itr + 1):
        solutions = scheduler.ask()

        (
            all_repaired,
            all_objective,
            all_measures,
            all_task_id,
        ) = ([], [], [], [])
        for eid, em in enumerate(scheduler.emitters):
            # Each emitter emits batch_size solutions
            sol_start = eid * cfg.envgen.emitter.batch_size
            sol_end = sol_start + cfg.envgen.emitter.batch_size
            repaired, objective, measures, trajectories = evaluators[
                em.task_id
            ].evaluate(solutions=solutions[sol_start:sol_end])

            all_repaired.extend(repaired)
            all_objective.extend(objective)
            all_measures.extend(measures)
            all_task_id += [em.task_id] * cfg.envgen.emitter.batch_size

            # Saves successful trajectories to the temporary dataset if they
            # come from an environment whose rollout success rate entropy > 0
            for entropy, env_rollouts in zip(objective, trajectories):
                if entropy > 0.01:
                    for traj in env_rollouts:
                        if traj.success:
                            temp_succ_dataset.write_episode(trajectory=traj)
                        else:
                            temp_fail_dataset.write_episode(trajectory=traj)

        all_repaired = np.array(all_repaired)
        all_objective = np.array(all_objective)
        all_measures = np.array(all_measures)
        all_task_id = np.array(all_task_id)

        edit_dists = np.linalg.norm(all_repaired - solutions, axis=1)
        # Solutions that have been modified by external repair are no longer
        # strictly emitted by CMA-ES and are injected into its update.
        injected = edit_dists > 0
        logger.info(
            f"{np.sum(injected)}/{injected.shape[0]} solutions have been modified by external repair"
        )
        logger.info(f"\nedit_dists={edit_dists}")

        scheduler.tell(
            # Penalize objective with MILP editing distance if there is any
            np.clip(all_objective - edit_dists, a_min=1e-6, a_max=None),
            all_measures,
            solution=all_repaired,
            injected=injected,
            task_id=all_task_id,
        )

        logger.info(
            f"------------------------ Iteration{i} ------------------------\n"
            f"\t QD-Score: {scheduler.archive.stats.qd_score}\n"
            f"\t Coverage: {scheduler.archive.stats.coverage}\n"
            f"\t Maximum : {scheduler.archive.stats.obj_max}\n"
            f"\t Average : {scheduler.archive.stats.obj_mean}\n"
        )

        final_itr = i == search_itr
        if i % cfg.log_every == 0 or final_itr:
            pkl.dump(
                scheduler,
                open(logdir / f"scheduler_{i:08d}.pkl", "wb"),
            )

            with open(summary_filename, "a") as summary_file:
                writer = csv.writer(summary_file)
                data = [
                    i,
                    scheduler.archive.stats.qd_score,
                    scheduler.archive.stats.coverage,
                    scheduler.archive.stats.obj_max,
                    scheduler.archive.stats.obj_mean,
                ]
                writer.writerow(data)

            if isinstance(scheduler.archive, GridArchive):
                save_heatmap(
                    scheduler.archive,
                    logdir / f"heatmap_{i:08d}.png",
                )

    # After qd loop finishes, exports all eligible trajectories to a lerobot
    # dataset
    temp_succ_dataset.convert_to_lerobot(cfg.lerobot_dataset_repo_id)


if __name__ == "__main__":
    main()
