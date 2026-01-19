from pathlib import Path

import hydra
import numpy as np
import tqdm
from dask.distributed import Client, LocalCluster
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.dataset_utils import TempDataset
from src.libero_spatial_eval import get_default_env_params
from src.qd_search import add_omegaconf_resolvers, seed_everything


@hydra.main(
    version_base=None, config_path="../config", config_name="collect_embeddings"
)
def collect_embeddings(cfg: DictConfig):
    add_omegaconf_resolvers()
    seed_everything(cfg.seed)

    embedding_dataset = TempDataset(dataset_dir=Path(cfg.dataset_dir))

    default_params = np.array(
        [get_default_env_params(tid) for tid in cfg.eval.task_ids]
    )

    archive = instantiate(
        cfg.envgen.archive,
        seed_solutions=default_params,
        extra_fields={
            "task_id": (
                (),
                np.int32,
            ),  # store task_id for each env for later eval
        },
    )

    emitters = [
        instantiate(
            cfg.envgen.emitter,
            archive=archive,
            x0=default_params[tid],
        )
        for tid in cfg.eval.task_ids
    ]
    for em, tid in zip(emitters, cfg.eval.task_ids):
        em.task_id = tid

    scheduler = instantiate(
        cfg.envgen.scheduler, archive=archive, emitters=emitters
    )

    if cfg.single_process:
        client = None
    else:
        cluster = LocalCluster(
            processes=True,
            n_workers=cfg.envgen.emitter.batch_size,
            threads_per_worker=1,
        )
        client = Client(cluster)

    evaluators = [
        instantiate(cfg.eval.task_eval, task_id=tid, dask_client=client)
        for tid in range(10)
    ]

    for _ in tqdm.trange(
        0,
        cfg.dataset_size,
        len(scheduler.emitters) * cfg.envgen.emitter.batch_size,
    ):
        solutions = scheduler.ask()

        all_task_id = []
        for eid, em in enumerate(scheduler.emitters):
            sol_start = eid * cfg.envgen.emitter.batch_size
            sol_end = sol_start + cfg.envgen.emitter.batch_size
            trajectories = evaluators[em.task_id].get_single_trajectories(
                solutions=solutions[sol_start:sol_end]
            )

            for traj in trajectories:
                embedding_dataset.write_episode(trajectory=traj)

            all_task_id += [em.task_id] * cfg.envgen.emitter.batch_size

        # Dummy objective and measure values
        scheduler.tell(
            np.repeat(1e-6, solutions.shape[0]),
            np.random.rand(solutions.shape[0], 1),
            solution=solutions,
            injected=np.full(solutions.shape[0], False),
            task_id=all_task_id,
        )


if __name__ == "__main__":
    collect_embeddings()
