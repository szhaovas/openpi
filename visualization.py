import pickle as pkl
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, no_update
from dask.distributed import Client, LocalCluster
from ribs.archives import ArchiveBase, GridArchive
from ribs.visualize import grid_archive_heatmap
from tqdm import tqdm

from src.dataset_utils import TempDataset
from src.env_search import (
    extract_scheduler_nevals,
    safe_pickle_dump,
    save_heatmap,
)
from src.libero_spatial_eval import LiberoSpatialEval
from src.measures import PCAMeasure


def show_interactive_archive(
    archive: GridArchive,
    vla_server_uris: List[str],
    logdir: str = "rollouts",
):
    if len(vla_server_uris) > 1:
        cluster = LocalCluster(
            processes=True,
            n_workers=len(vla_server_uris),
            threads_per_worker=1,
        )
        client = Client(cluster)
    else:
        client = None

    evaluators = [
        LiberoSpatialEval(
            task_id=tid,
            objective_func="success_rate",
            num_trials_per_sol=len(vla_server_uris),
            vla_server_uris=vla_server_uris,
            dask_client=client,
            camera_heights=256,
            camera_widths=256,
            # replan_steps=1,  # Uncomment when no action chunking
            repair_config={"time_limit": 1500, "seed": 42},
        )
        for tid in range(10)
    ]

    fig = plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=1, cmap="viridis")
    plt.tight_layout()

    def onclick(event):
        occupied, data = archive.retrieve_single([event.xdata, event.ydata])

        if occupied:
            print(
                f"Recorded objective: {data['objective']}; Recorded measures: {data['measures']}"
            )
            _, objective, _, trajectories = evaluators[
                data["task_id"]
            ].evaluate_single(solution=data["solution"])
            print(f"Evaluated success rate: {objective}")

            for traj in trajectories:
                rollout_dataset = TempDataset(dataset_dir=Path(logdir))
                rollout_dataset.write_episode(trajectory=traj)
        else:
            print("Archive cell not occupied")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


def _plotly_grid_archive_heatmap(
    archive: GridArchive, shape: Dict[str, int] = {"width": 600, "height": 600}
) -> go.Figure:
    x_dim, y_dim = archive.dims
    colors = np.full((y_dim, x_dim), np.nan)
    index_batch = archive.data("index")
    objective_batch = archive.data("objective")
    grid_index_batch = archive.int_to_grid_index(index_batch)
    colors[grid_index_batch[:, 1], grid_index_batch[:, 0]] = objective_batch

    x_bounds = archive.boundaries[0]
    y_bounds = archive.boundaries[1]

    fig = go.Figure(
        data=go.Heatmap(
            z=colors,
            x=x_bounds,
            y=y_bounds,
            colorbar={"title": "Ent. success"},
            colorscale="Viridis",
        )
    )
    fig.update_layout(**shape)
    fig.update_xaxes(title="Spread")
    fig.update_yaxes(title="Similarity")

    return fig


def host_interactive_archive(
    archive: GridArchive,
    vla_server_uris: List[str],
    port: int = 8050,
    logdir: str = "rollouts",
):
    """Similar to :func:`show_interactive_archive`, except it hosts the
    interactive plot at localhost:<port> to allow generating the plot on a
    remote machine and then viewing it on your local machine (e.g. if you only
    have access to ssh). After configuring port forwarding between your local
    machine and ``port`` on the remote machine, you will be able to view and
    interact with the plot on your local machine's browser.

    Args:
        archive (GridArchive): Archive to be displayed.
        port (int): The port on which to display the plot.
    """
    if len(vla_server_uris) > 1:
        cluster = LocalCluster(
            processes=True,
            n_workers=len(vla_server_uris),
            threads_per_worker=1,
        )
        client = Client(cluster)
    else:
        client = None

    evaluators = [
        LiberoSpatialEval(
            task_id=tid,
            objective_func="success_rate",
            num_trials_per_sol=len(vla_server_uris),
            vla_server_uris=vla_server_uris,
            dask_client=client,
            camera_heights=256,
            camera_widths=256,
            # replan_steps=1,  # Uncomment when no action chunking
            repair_config={"time_limit": 1500, "seed": 42},
        )
        for tid in range(10)
    ]

    app = Dash(__name__)

    app.layout = html.Div(
        [
            dcc.Graph(
                id="archive-heatmap",
                figure=_plotly_grid_archive_heatmap(archive),
            ),
            html.Div(
                id="dummy-output", style={"display": "none"}
            ),  # hidden dummy output
        ],
        style={"display": "flex", "justifyContent": "center"},
    )

    @app.callback(
        Output("dummy-output", "children"),
        Input("archive-heatmap", "clickData"),
    )
    def onclick(clickData):
        if clickData is None:
            return no_update

        occupied, data = archive.retrieve_single(
            [clickData["points"][0]["x"], clickData["points"][0]["y"]]
        )

        if occupied:
            print(
                f"Recorded objective: {data['objective']}; Recorded measures: {data['measures']}"
            )
            _, objective, _, trajectories = evaluators[
                data["task_id"]
            ].evaluate_single(solution=data["solution"])
            print(f"Evaluated success rate: {objective}")

            for traj in trajectories:
                rollout_dataset = TempDataset(dataset_dir=Path(logdir))
                rollout_dataset.write_episode(trajectory=traj)
        else:
            print("Archive cell not occupied")

        return None

    app.run(host="0.0.0.0", port=port, debug=True)


def success_rates_on_envs(
    env_archive: ArchiveBase,
    vla_server_uris: List[str],
    logdir: str = "success_rates",
):
    logpath = Path(logdir)
    assert not logpath.is_dir()

    if len(vla_server_uris) > 1:
        cluster = LocalCluster(
            processes=True,
            n_workers=len(vla_server_uris),
            threads_per_worker=1,
        )
        client = Client(cluster)
    else:
        client = None

    evaluators = [
        LiberoSpatialEval(
            task_id=tid,
            objective_func="success_rate",
            num_trials_per_sol=len(vla_server_uris),
            vla_server_uris=vla_server_uris,
            dask_client=client,
            camera_heights=256,
            camera_widths=256,
            # replan_steps=1,  # Uncomment when no action chunking
            repair_config={"time_limit": 1500, "seed": 42},
        )
        for tid in range(10)
    ]

    num_envs_per_task = len(env_archive) // 10
    success_rates_archive = GridArchive(
        solution_dim=env_archive.solution_dim,
        dims=[10, num_envs_per_task],
        ranges=[[0, 10], [0, num_envs_per_task]],
    )

    env_counters = [0] * 10
    for cell in tqdm(env_archive):
        task_id = cell["task_id"]
        env_id = env_counters[task_id]

        rollout_dataset = TempDataset(
            dataset_dir=logpath / f"task{task_id}_env{env_id}"
        )

        _, objective, _, trajectories = evaluators[task_id].evaluate_single(
            solution=cell["solution"]
        )

        success_rates_archive.add_single(
            solution=cell["solution"],
            objective=objective,
            measures=[task_id, env_id],
        )

        env_counters[task_id] += 1

        tqdm.write(f"task{task_id}_env{env_id} success rate: {objective}")

        for traj in trajectories:
            rollout_dataset.write_episode(trajectory=traj)

        safe_pickle_dump(success_rates_archive, logpath / "success_rates.pkl")

        save_heatmap(success_rates_archive, str(logpath / "success_rates.png"))


def redraw_heatmap(
    experiment_logdir: str,
    encoder_ckpt_path: str,
):
    measure_model = PCAMeasure(ckpt_path=encoder_ckpt_path)

    all_nevals = extract_scheduler_nevals(experiment_logdir)

    all_qd_scores = []
    all_coverages = []
    for filename, nevals in all_nevals.items():
        print(filename)
        with open(file=filename, mode="rb") as f:
            archive = pkl.load(f).archive

        dummy_archive = GridArchive(
            solution_dim=archive.solution_dim,
            dims=[100, 100],
            ranges=[[0, 1], [0, 1]],
        )

        Z = measure_model.model.transform(archive.data("embedding"))
        measures = (
            Z - (measure_model.lb - 0.5 * (measure_model.ub - measure_model.lb))
        ) / (2 * (measure_model.ub - measure_model.lb))

        dummy_archive.add(
            solution=archive.data("solution"),
            objective=archive.data("objective"),
            measures=measures,
        )

        all_qd_scores.append(dummy_archive.stats.qd_score)
        all_coverages.append(dummy_archive.stats.coverage)

        # save_heatmap(
        #     dummy_archive, f"{experiment_logdir}/heatmap_{nevals:08d}.png"
        # )
    all_qd_scores = np.array(all_qd_scores)
    all_qd_scores = all_qd_scores[np.argsort(list(all_nevals.values()))]
    print(all_qd_scores)
    all_coverages = np.array(all_coverages)
    all_coverages = all_coverages[np.argsort(list(all_nevals.values()))]
    print(all_coverages)


def tally_traj_by_task(
    traj_dataset: TempDataset, max_traj_len: Optional[int] = None
) -> Dict[str, List]:
    traj_id_by_task = {
        "pick the akita black bowl from table center and place it on the plate": [],
        "pick the akita black bowl next to the plate and place it on the plate": [],
        "pick the akita black bowl on the ramekin and place it on the plate": [],
        "pick the akita black bowl next to the ramekin and place it on the plate": [],
        "pick the akita black bowl on the cookies box and place it on the plate": [],
        "pick the akita black bowl on the stove and place it on the plate": [],
        "pick the akita black bowl next to the cookies box and place it on the plate": [],
        "pick the akita black bowl between the plate and the ramekin and place it on the plate": [],
        "pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate": [],
        "pick the akita black bowl on the wooden cabinet and place it on the plate": [],
    }
    for traj_id, traj in enumerate(tqdm(traj_dataset)):
        assert traj.prompt is not None
        if max_traj_len is None or len(traj.state) < max_traj_len:
            traj_id_by_task[traj.prompt].append(traj_id)

    return traj_id_by_task


if __name__ == "__main__":
    # with open(
    #     file="outputs/domain_randomization/2026-02-07_003429/embeddings/test_scenarios.pkl",
    #     mode="rb",
    # ) as f:
    #     env_archive = pkl.load(f)
    #     success_rates_on_envs(
    #         env_archive=env_archive,
    #         vla_server_uris=[
    #             "0.0.0.0:8001",
    #             "0.0.0.0:8002",
    #             "0.0.0.0:8003",
    #             "0.0.0.0:8004",
    #         ],
    #         logdir="vlarl_libero",
    #     )

    with open(
        file="results/domain_randomization/scheduler_00002304.pkl",
        mode="rb",
    ) as f:
        archive = pkl.load(f).archive
        host_interactive_archive(
            archive,
            vla_server_uris=[
                "0.0.0.0:8001",
                "0.0.0.0:8002",
                "0.0.0.0:8003",
                "0.0.0.0:8004",
            ],
            logdir="rollouts_base",
        )
