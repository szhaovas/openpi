import csv
import glob
import logging
import os
import pickle as pkl
import random
import re
import shutil
from functools import reduce
from operator import add, mul
from pathlib import Path
from typing import Any, Dict, Optional, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from dask.distributed import Client, LocalCluster
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ribs.archives import ArchiveBase, CVTArchive, GridArchive
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tqdm
from vendi_score import vendi

from src.dataset_utils import (
    TempDataset,
    Trajectory,
    filter_lerobot_dataset_by_task,
)
from src.libero_spatial_eval import (
    get_default_env_params,
    libero_spatial_prompts,
)

logger = logging.getLogger(__name__)


def _monkey_patch_pkl_load():
    """Run this at the beginning of main if you see an error with
    __generator_ctor expecting 1 argument when loading from a pkl checkpoint.
    """
    import importlib

    mod = importlib.import_module("numpy.random._pickle")
    orig = getattr(mod, "__generator_ctor", None)

    def compat_generator_ctor(*args, **kwargs):
        # Try to be permissive: if orig accepts the args, call it
        try:
            return orig(*args, **kwargs)
        except TypeError:
            # If orig expects a single arg (the more common case), call with first arg
            if len(args) >= 1:
                return orig(args[0])
            # fallback: raise original error
            raise

    if orig is not None:
        mod.__generator_ctor = compat_generator_ctor


def extract_scheduler_nevals(experiment_logdir: str) -> Dict[str, int]:
    """Extracts the numbers of evaluations at all scheduler checkpoints within
    an experiment log. Returns a dictionary matching checkpoint filenames with
    extracted numbers of evaluations.

    A scheduler checkpoint is expected to be named after the
    ``scheduler_[0-9]{8}.pkl`` format, in which the digits record its number
    of evaluations.
    """
    all_scheduler_ckpt = glob.glob(
        f"{experiment_logdir}/scheduler_{'[0-9]'*8}.pkl"
    )

    result = {}
    pattern = r"scheduler_(\d{8})\.pkl"
    for filename in all_scheduler_ckpt:
        match = re.search(pattern, filename)
        if match:
            result[filename] = int(match.group(1))

    return result


def _delete_trajectories_after_idx(dataset: TempDataset, del_after: int):
    """Deletes all trajectories in ``dataset`` whose indices >= ``del_after``."""
    for idx in range(del_after, len(dataset)):
        eps_dir = dataset.dataset_dir / f"ep_{idx:05d}"
        shutil.rmtree(eps_dir)


def _filter_succ_dataset(
    env_archive: ArchiveBase,
    succ_dataset: TempDataset,
    filtered_dataset_path: Path,
    max_traj_len: Optional[int] = None,
) -> TempDataset:
    """``succ_dataset`` may contain multiple rollout trajectories for each
    environment stored in ``env_archive``, which can cause action divergence.
    This function filters ``succ_dataset`` so that it contains only a single
    shortest (which implies highest-quality) trajectory for each environment.

    Args:
        env_archive: A QD archive that contains environmental parameters found
            by the env_search pipeline. It must also contain a ``succ_traj_idx``
            extra field mapping each environment to some trajectories in
            ``succ_dataset``.
        succ_dataset: Rollout trajectory dataset containing successful
            trajectories rolled out on ``env_archive`` environments.
        filtered_dataset_path: The filtered dataset will be saved to this path.
        max_traj_len: If specified, only trajectories shorter than this
            threshold will be saved to the filtered dataset.

    Returns:
        filtered_succ_dataset
    """
    succ_traj_idx = env_archive.data("succ_traj_idx")
    assert succ_traj_idx is not None

    filtered_succ_dataset = TempDataset(dataset_dir=filtered_dataset_path)

    for same_env_traj_idx in tqdm(
        succ_traj_idx, desc="Filtering collected trajs"
    ):
        shortest_traj = None
        min_len = np.inf
        for traj_idx in same_env_traj_idx[same_env_traj_idx != -1]:
            traj = succ_dataset[traj_idx]
            if len(traj.state) < min_len:
                shortest_traj = traj
                min_len = len(traj.state)

        if shortest_traj is not None and (
            max_traj_len is None or min_len < max_traj_len
        ):
            filtered_succ_dataset.write_episode(trajectory=shortest_traj)

    return filtered_succ_dataset


def safe_pickle_dump(obj: Any, save_path: Path):
    """Checks free disk space before picking an object. This prevents using up
    all remaining disk space with a partially-written and unusable object.
    """
    obj_size = len(pkl.dumps(obj, protocol=pkl.HIGHEST_PROTOCOL))
    free_space = shutil.disk_usage(save_path.parent).free

    if obj_size > free_space:
        raise OSError(
            f"Not enough disk space: need {obj_size} bytes, have {free_space}"
        )

    with open(save_path, "wb") as f:
        pkl.dump(obj, f)


def save_heatmap(
    archive: Union[GridArchive, CVTArchive],
    heatmap_path: str,
    vmin: int = 0,
    vmax: int = 1,
):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    if isinstance(archive, GridArchive):
        grid_archive_heatmap(archive, vmin=vmin, vmax=vmax, cmap="viridis")
    elif isinstance(archive, CVTArchive):
        cvt_archive_heatmap(archive, vmin=vmin, vmax=vmax, cmap="viridis")
    else:
        raise ValueError
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def add_omegaconf_resolvers():
    OmegaConf.register_new_resolver(
        "oc.env_int", lambda name: int(os.environ[name])
    )
    OmegaConf.register_new_resolver(
        "oc.env_csv", lambda name: os.environ[name].split(",")
    )
    OmegaConf.register_new_resolver("oc.equals", lambda x, y: x == y)
    OmegaConf.register_new_resolver("oc.len", lambda x: len(x))
    OmegaConf.register_new_resolver(
        "oc.add", lambda *args: reduce(add, args, 1)
    )
    OmegaConf.register_new_resolver(
        "oc.mul", lambda *args: reduce(mul, args, 1)
    )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_embeddings(colemb_cfg: DictConfig):
    embedding_logdir = Path(colemb_cfg.save_embeddings_to)

    embedding_dataset = TempDataset(dataset_dir=embedding_logdir)

    archive = instantiate(
        colemb_cfg.envgen.logging_archive,
        solution_dim=len(get_default_env_params()),
        extra_fields={
            "task_id": (
                (),
                np.int32,
            ),  # store task_id for each env for later eval
        },
    )

    emitters = [
        instantiate(
            colemb_cfg.envgen.emitter,
            archive=archive,
            sigma=colemb_cfg.collect_embedding_sigma,
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

    with tqdm(range(colemb_cfg.collect_embedding_num_evals)) as pbar:
        while pbar.n < pbar.total:
            solutions = scheduler.ask()

            all_task_id = []
            for eid, em in enumerate(scheduler.emitters):
                sol_start = eid * colemb_cfg.envgen.emitter.batch_size
                sol_end = sol_start + colemb_cfg.envgen.emitter.batch_size
                repaired, trajectories = evaluators[
                    em.task_id
                ].get_single_trajectories(
                    solutions=solutions[sol_start:sol_end]
                )
                pbar.update(len(trajectories))
                pbar.refresh()

                all_task_id += [
                    em.task_id
                ] * colemb_cfg.envgen.emitter.batch_size

                for traj in trajectories:
                    embedding_dataset.write_episode(trajectory=traj)

            # Dummy objective and measure values
            scheduler.tell(
                np.repeat(1e-6, solutions.shape[0]),
                np.random.rand(solutions.shape[0], 1),
                solution=repaired,
                injected=np.full(solutions.shape[0], False),
                task_id=all_task_id,
            )

    safe_pickle_dump(scheduler.archive, embedding_logdir / "test_envs.pkl")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    add_omegaconf_resolvers()
    seed_everything(cfg.seed)
    # _monkey_patch_pkl_load()  # TODO: Remove when publishing

    if cfg.reload_from_dir is None:
        logdir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
        summary_filename = logdir / "summary.csv"
        starting_nevals = 0

        temp_succ_dataset = TempDataset(dataset_dir=logdir / "succ_dataset")
        temp_fail_dataset = TempDataset(dataset_dir=logdir / "fail_dataset")

        # QD archive for QD visualization and metrics.
        # In the case of cma-mae, also provides x0 in case of restart.
        qd_archive = instantiate(
            cfg.envgen.qd_archive,
            solution_dim=len(get_default_env_params()),
            extra_fields={
                "task_id": (
                    (),
                    np.int32,
                ),  # store task_id for each env for later eval
                "embedding": (
                    (cfg.eval.measure_model.model_cfg.input_dim,),
                    np.float32,
                ),
                "succ_traj_idx": (
                    (cfg.eval.task_eval.num_trials_per_sol,),
                    np.int32,
                ),  # for retrieving successful rollouts on each environment
                "fail_traj_idx": (
                    (cfg.eval.task_eval.num_trials_per_sol,),
                    np.int32,
                ),  # for retrieving failed rollouts on each environment
            },
        )

        # For saving all results found throughout the search.
        # This is needed because QD archive implements elitism and may discard past results if better alternatives are found.
        logging_archive = instantiate(
            cfg.envgen.logging_archive,
            solution_dim=len(get_default_env_params()),
            extra_fields={
                "task_id": (
                    (),
                    np.int32,
                ),  # store task_id for each env for later eval
                "embedding": (
                    (cfg.eval.measure_model.model_cfg.input_dim,),
                    np.float32,
                ),
                "succ_traj_idx": (
                    (cfg.eval.task_eval.num_trials_per_sol,),
                    np.int32,
                ),  # for retrieving successful rollouts on each environment
                "fail_traj_idx": (
                    (cfg.eval.task_eval.num_trials_per_sol,),
                    np.int32,
                ),  # for retrieving failed rollouts on each environment
            },
        )

        # Emitters for generating noisy environments
        # Each task_id gets a single emitter whose x0 is set to its
        # corresponding default.
        emitters = [
            instantiate(
                cfg.envgen.emitter,
                archive=qd_archive,
                x0=get_default_env_params(tid),
            )
            for tid in cfg.eval.task_ids
        ]
        for em, tid in zip(emitters, cfg.eval.task_ids):
            em.task_id = tid

        # Scheduler for switching among tasks
        scheduler = instantiate(
            cfg.envgen.scheduler,
            archive=qd_archive,
            emitters=emitters,
            result_archive=logging_archive,
        )

        # Collect some embeddings to train a measure model for latent representations
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
        measure_model = instantiate(cfg.eval.measure_model)

        with open(summary_filename, "w") as summary_file:
            writer = csv.writer(summary_file)
            writer.writerow(
                [
                    "Num.Evals",
                    "QD-Score",
                    "Coverage",
                    "Maximum",
                    "Average",
                    "Num.Envs",
                    "Avg.EmbDist",
                    "Vendi-Score",
                    "QVendi-Score",
                ]
            )
    else:
        logdir = Path(cfg.reload_from_dir)
        summary_filename = Path(cfg.reload_from_dir) / "summary.csv"
        starting_nevals = max(
            extract_scheduler_nevals(cfg.reload_from_dir).values()
        )

        temp_succ_dataset = TempDataset(dataset_dir=logdir / "succ_dataset")
        temp_fail_dataset = TempDataset(dataset_dir=logdir / "fail_dataset")

        with open(
            file=logdir / f"scheduler_{starting_nevals:08d}.pkl",
            mode="rb",
        ) as f:
            scheduler = pkl.load(f)

            # clean up orphan trajectories whose rollout environments are not
            # stored in the latest checkpoint
            latest_succ_dataset_idx = np.max(
                scheduler.result_archive.data("succ_traj_idx")
            )
            logger.warning(
                f"Envs in latest checkpoint map to successful trajectory idx up to {latest_succ_dataset_idx}, "
                f"deleting orphan successful trajectories with idx >= {latest_succ_dataset_idx+1}..."
            )
            _delete_trajectories_after_idx(
                temp_succ_dataset, latest_succ_dataset_idx + 1
            )
            latest_fail_dataset_idx = np.max(
                scheduler.result_archive.data("fail_traj_idx")
            )
            logger.warning(
                f"Envs in latest checkpoint map to failed trajectory idx up to {latest_fail_dataset_idx}, "
                f"deleting orphan failed trajectories with idx >= {latest_fail_dataset_idx+1}..."
            )
            _delete_trajectories_after_idx(
                temp_fail_dataset, latest_fail_dataset_idx + 1
            )

        measure_model = instantiate(
            cfg.eval.measure_model,
            ckpt_path=logdir / "measure_ckpt.pt",
        )

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
            measure_model=measure_model,
        )
        for tid in range(10)
    ]

    nevals_since_last_log = 0
    with tqdm(
        range(starting_nevals, cfg.env_search_num_evals),
        initial=starting_nevals,
        total=cfg.env_search_num_evals,
    ) as pbar:
        while pbar.n < pbar.total:
            solutions = scheduler.ask()

            (
                all_repaired,
                all_objective,
                all_measures,
                all_task_id,
                all_embedding,
                all_succ_traj_idx,
                all_fail_traj_idx,
            ) = ([], [], [], [], [], [], [])
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

                for env_rollouts in trajectories:
                    all_embedding.append(
                        np.mean(
                            [
                                traj.embedding[
                                    0
                                ]  # first embedding before any action
                                for traj in env_rollouts
                                if len(traj.embedding) > 0
                            ],
                            axis=0,
                        )
                    )

                # Saves successful trajectories to the temporary dataset if
                # they come from a challenging yet feasible environment on
                # which there are some successful and some failed rollouts
                for env_rollouts in trajectories:
                    succ_traj_idx = np.full(
                        cfg.eval.task_eval.num_trials_per_sol,
                        -1,
                        dtype=np.int32,
                    )
                    fail_traj_idx = np.full(
                        cfg.eval.task_eval.num_trials_per_sol,
                        -1,
                        dtype=np.int32,
                    )

                    if np.any(
                        [traj.success for traj in env_rollouts]
                    ) and np.any([not traj.success for traj in env_rollouts]):
                        stid, ftid = [], []
                        for traj in env_rollouts:
                            if traj.success:
                                stid.append(
                                    temp_succ_dataset.write_episode(
                                        trajectory=traj
                                    )
                                )
                            else:
                                ftid.append(
                                    temp_fail_dataset.write_episode(
                                        trajectory=traj
                                    )
                                )

                        succ_traj_idx[: len(stid)] = stid
                        fail_traj_idx[: len(ftid)] = ftid

                    all_succ_traj_idx.append(succ_traj_idx)
                    all_fail_traj_idx.append(fail_traj_idx)

            all_repaired = np.array(all_repaired)
            all_objective = np.array(all_objective)
            all_measures = np.array(all_measures)
            all_task_id = np.array(all_task_id)
            all_embedding = np.array(all_embedding)
            all_succ_traj_idx = np.array(all_succ_traj_idx)
            all_fail_traj_idx = np.array(all_fail_traj_idx)

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
                succ_traj_idx=all_succ_traj_idx,
                fail_traj_idx=all_fail_traj_idx,
            )

            # These metrics need to be computed over all historical results. Use logging_archive
            archive_data = scheduler.result_archive.data(
                ["objective", "embedding"]
            )
            feasible_env_idx = np.where(archive_data["objective"] > 1e-6)[0]
            num_feasible_envs = len(feasible_env_idx)

            if num_feasible_envs > 0:
                avg_emb_dist = np.mean(
                    pairwise_distances(
                        X=archive_data["embedding"][feasible_env_idx],
                        metric="euclidean",
                    )
                )
                rbf_K = rbf_kernel(
                    X=archive_data["embedding"][feasible_env_idx], gamma=0.002
                )  # gamma is chosen to match median embedding distance
                emb_vendi = vendi.score_K(rbf_K, normalize=True)
            else:
                avg_emb_dist, emb_vendi = 0, 0
            emb_qvendi = np.mean(archive_data["objective"]) * emb_vendi

            # Use qd_archive for QD metrics
            logger.info(
                f"---------------------- Eval {pbar.n:08d} ----------------------\n"
                f"\t QD-Score     : {scheduler.archive.stats.qd_score}\n"
                f"\t Coverage     : {scheduler.archive.stats.coverage}\n"
                f"\t Maximum      : {scheduler.archive.stats.obj_max}\n"
                f"\t Average      : {scheduler.archive.stats.obj_mean}\n"
                f"\t Num.Envs     : {num_feasible_envs}\n"
                f"\t Avg.EmbDist  : {avg_emb_dist}\n"
                f"\t Vendi-Score  : {emb_vendi}\n"
                f"\t QVendi-Score : {emb_qvendi}\n"
            )

            final_itr = pbar.n == pbar.total
            if nevals_since_last_log > cfg.log_every or final_itr:
                nevals_since_last_log = 0

                safe_pickle_dump(
                    scheduler, logdir / f"scheduler_{pbar.n:08d}.pkl"
                )

                with open(summary_filename, "a") as summary_file:
                    writer = csv.writer(summary_file)
                    data = [
                        pbar.n,
                        scheduler.archive.stats.qd_score,
                        scheduler.archive.stats.coverage,
                        scheduler.archive.stats.obj_max,
                        scheduler.archive.stats.obj_mean,
                        num_feasible_envs,
                        avg_emb_dist,
                        emb_vendi,
                        emb_qvendi,
                    ]
                    writer.writerow(data)

                # Use qd_archive for QD heatmap
                if scheduler.archive.measure_dim <= 2:
                    save_heatmap(
                        scheduler.archive, logdir / f"heatmap_{pbar.n:08d}.png"
                    )

    # Exports finetune dataset after qd search finishes
    # Filters out redundant and low-quality trajectories
    logger.info("Env search done! Generating finetune dataset...")
    finetune_dataset = _filter_succ_dataset(
        env_archive=scheduler.result_archive,
        succ_dataset=temp_succ_dataset,
        filtered_dataset_path=logdir / "finetune_dataset",
        max_traj_len=110,
    )

    # Adds some trajectories from the base libero dataset to prevent forgetting
    filtered_lerobot_dataset = filter_lerobot_dataset_by_task(
        repo_id="physical-intelligence/libero",
        task_prompts=libero_spatial_prompts[cfg.eval.task_ids],
    )
    for traj_start, traj_end in tqdm(
        zip(
            filtered_lerobot_dataset.episode_data_index["from"].numpy(),
            filtered_lerobot_dataset.episode_data_index["to"].numpy(),
        ),
        total=len(filtered_lerobot_dataset.episode_data_index["from"]),
        desc="Adding base trajs",
    ):
        if np.random.rand() > cfg.proportion_of_base_data_to_mix:
            continue

        trajectory = Trajectory(
            prompt=filtered_lerobot_dataset[int(traj_start)]["task"],
            success=True,
        )
        for step_id in range(traj_start, traj_end):
            trajectory.image.append(
                np.transpose(
                    filtered_lerobot_dataset[step_id]["image"].numpy() * 255,
                    (1, 2, 0),  # chw -> hwc
                ).astype("uint8")
            )
            trajectory.wrist_image.append(
                np.transpose(
                    filtered_lerobot_dataset[step_id]["wrist_image"].numpy()
                    * 255,
                    (1, 2, 0),  # chw -> hwc
                ).astype("uint8")
            )
            trajectory.state.append(
                filtered_lerobot_dataset[step_id]["state"].numpy()
            )
            trajectory.action.append(
                filtered_lerobot_dataset[step_id]["actions"].numpy()
            )

        finetune_dataset.write_episode(trajectory)

    # Convert to finetuning format
    # finetune_dataset.convert_to_lerobot(cfg.output_dataset_id)
    finetune_dataset.convert_to_rlds(cfg.output_dataset_id)


if __name__ == "__main__":
    main()
