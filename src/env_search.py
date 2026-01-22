import csv
import glob
import logging
import os
import pickle as pkl
import re
from functools import reduce
from operator import add, mul
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from dask.distributed import Client, LocalCluster
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap
from scipy.spatial.distance import pdist

from src.dataset_utils import TempDataset
from src.libero_spatial_eval import get_default_env_params

logger = logging.getLogger(__name__)


def _extract_reload_nevals(reload_dir: str) -> int:
    """Extracts the number of evaluations from an experiment log folder by
    finding the latest scheduler checkpoint and reading its number of
    evaluations. A scheduler checkpoint is expected to be named after the
    ``scheduler_[0-9]{8}.pkl`` format, in which the digits record its number
    of evaluations. If unable to find such a checkpoint, returns 0.
    """
    all_scheduler_ckpt = glob.glob(f"{reload_dir}/scheduler_{'[0-9]'*8}.pkl")

    reload_nevals = 0
    pattern = r"scheduler_(\d{8})\.pkl"
    for filename in all_scheduler_ckpt:
        match = re.search(pattern, filename)
        if match:
            reload_nevals = max(reload_nevals, int(match.group(1)))

    if reload_nevals == 0:
        logger.warning("_extract_reload_nevals failed, returning 0")

    return reload_nevals


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
    embedding_dataset = TempDataset(
        dataset_dir=Path(colemb_cfg.save_embeddings_to)
    )

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
            n_workers=int(os.environ["NUM_SERVERS"]),
            threads_per_worker=1,
        )
        client = Client(cluster)

    evaluators = [
        instantiate(colemb_cfg.eval.task_eval, task_id=tid, dask_client=client)
        for tid in range(10)
    ]

    with tqdm.tqdm(
        range(1, colemb_cfg.collect_embedding_num_evals + 1)
    ) as pbar:
        while pbar.n < pbar.total:
            solutions = scheduler.ask()

            for eid, em in enumerate(scheduler.emitters):
                sol_start = eid * colemb_cfg.envgen.emitter.batch_size
                sol_end = sol_start + colemb_cfg.envgen.emitter.batch_size
                trajectories = evaluators[em.task_id].get_single_trajectories(
                    solutions=solutions[sol_start:sol_end]
                )
                pbar.update(len(trajectories))
                pbar.refresh()

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

    if cfg.reload_from_dir is None:
        logdir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
        summary_filename = logdir / "summary.csv"
        starting_nevals = 1

        # Main archive for storing environments found by the pipelin
        archive = instantiate(
            cfg.envgen.archive,
            solution_dim=len(get_default_env_params()),
            extra_fields={
                "task_id": (
                    (),
                    np.int32,
                ),  # store task_id for each env for later eval
                "embedding": (
                    (cfg.eval.measure_encoder.model_cfg.input_dim,),
                    np.float32,
                ),
            },
        )

        # Passive archive for computing QD metrics
        result_archive_cfg = hydra.compose(
            config_name="main", overrides=["envgen=cma_mae"]
        ).envgen.archive
        result_archive = instantiate(
            result_archive_cfg,
            solution_dim=len(get_default_env_params()),
            learning_rate=None,
            threshold_min=-np.inf,
            extra_fields={
                "task_id": (
                    (),
                    np.int32,
                ),  # store task_id for each env for later eval
                "embedding": (
                    (cfg.eval.measure_encoder.model_cfg.input_dim,),
                    np.float32,
                ),
            },
        )

        # Emitters for generating noisy environments
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

        # Scheduler for switching among tasks
        scheduler = instantiate(
            cfg.envgen.scheduler,
            archive=archive,
            emitters=emitters,
            result_archive=result_archive,
        )

        # Collect some embeddings to train an autoencoder for latent representations
        # Not needed by domain randomization itself but still useful for logging and visualization
        collect_embedding_config = hydra.compose(
            config_name="main",
            overrides=[
                "envgen=domain_randomization",
                "eval=first_timestep",
                "single_process=true",
            ],
        )
        collect_embeddings(collect_embedding_config)
        encoder_manager = instantiate(cfg.eval.measure_encoder)

        with open(summary_filename, "w") as summary_file:
            writer = csv.writer(summary_file)
            writer.writerow(
                [
                    "Iteration",
                    "QD-Score",
                    "Coverage",
                    "Maximum",
                    "Average",
                    "Num.ValidEnv",
                    "Avg.EmbDist",
                ]
            )
            # Number of challenging environments
            # Average embedding distance
    else:
        logdir = Path(cfg.reload_from_dir)
        summary_filename = Path(cfg.reload_from_dir) / "summary.csv"
        starting_nevals = _extract_reload_nevals(cfg.reload_from_dir)

        with open(
            file=logdir / f"scheduler_{starting_nevals:08d}.pkl",
            mode="rb",
        ) as f:
            scheduler = pkl.load(f)

        encoder_manager = instantiate(
            cfg.eval.measure_encoder,
            ckpt_path=logdir / "encoder_ckpt.pt",
        )

    temp_succ_dataset = TempDataset(dataset_dir=logdir / "succ_dataset")
    temp_fail_dataset = TempDataset(dataset_dir=logdir / "fail_dataset")

    if cfg.single_process:
        client = None
    else:
        cluster = LocalCluster(
            processes=True,
            n_workers=cfg.eval.task_eval.num_trials_per_sol,
            threads_per_worker=1,
        )
        client = Client(cluster)

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

    nevals_since_last_log = 0
    with tqdm.tqdm(
        range(starting_nevals, cfg.env_search_num_evals + 1)
    ) as pbar:
        while pbar.n < pbar.total:
            solutions = scheduler.ask()

            (
                all_repaired,
                all_objective,
                all_measures,
                all_task_id,
                all_embedding,
            ) = ([], [], [], [], [])
            for eid, em in enumerate(scheduler.emitters):
                # Each emitter emits batch_size solutions
                sol_start = eid * cfg.envgen.emitter.batch_size
                sol_end = sol_start + cfg.envgen.emitter.batch_size
                repaired, objective, measures, trajectories = evaluators[
                    em.task_id
                ].evaluate(solutions=solutions[sol_start:sol_end])
                pbar.update(len(trajectories))
                pbar.refresh()
                nevals_since_last_log += len(trajectories)

                all_repaired.extend(repaired)
                all_objective.extend(objective)
                all_measures.extend(measures)
                all_task_id += [em.task_id] * cfg.envgen.emitter.batch_size

                for sol_rollouts in trajectories:
                    all_embedding.append(
                        np.mean(
                            [traj.embedding for traj in sol_rollouts], axis=0
                        )
                    )

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
            all_embedding = np.array(all_embedding)

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
                embedding=all_embedding,
            )

            archive_data = scheduler.result_archive.data(
                ["objective", "embedding"]
            )
            avg_emb_dist = np.mean(
                pdist(archive_data["embedding"], metric="euclidean")
            )
            num_valid_env = np.sum(archive_data["objective"] > 0.01)

            logger.info(
                f"---------------------- Eval {pbar.n:08d} ----------------------\n"
                f"\t QD-Score: {scheduler.result_archive.stats.qd_score}\n"
                f"\t Coverage: {scheduler.result_archive.stats.coverage}\n"
                f"\t Maximum : {scheduler.result_archive.stats.obj_max}\n"
                f"\t Average : {scheduler.result_archive.stats.obj_mean}\n"
                f"\t Num.ValidEnv : {num_valid_env}\n"
                f"\t Avg.EmbDist : {avg_emb_dist}\n"
            )

            final_itr = pbar.n == cfg.env_search_num_evals
            if nevals_since_last_log > cfg.log_every or final_itr:
                nevals_since_last_log = 0
                pkl.dump(
                    scheduler,
                    open(logdir / f"scheduler_{pbar.n:08d}.pkl", "wb"),
                )

                with open(summary_filename, "a") as summary_file:
                    writer = csv.writer(summary_file)
                    data = [
                        pbar.n,
                        scheduler.result_archive.stats.qd_score,
                        scheduler.result_archive.stats.coverage,
                        scheduler.result_archive.stats.obj_max,
                        scheduler.result_archive.stats.obj_mean,
                        num_valid_env,
                        avg_emb_dist,
                    ]
                    writer.writerow(data)

                if isinstance(scheduler.result_archive, GridArchive):
                    save_heatmap(
                        scheduler.result_archive,
                        logdir / f"heatmap_{pbar.n:08d}.png",
                    )

    # After qd loop finishes, exports all eligible trajectories to a lerobot
    # dataset
    temp_succ_dataset.convert_to_lerobot(cfg.lerobot_dataset_repo_id)


if __name__ == "__main__":
    main()
