import pickle as pkl
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import tqdm
from dash import Dash, Input, Output, dcc, html, no_update
from ribs.archives import ArchiveBase, GridArchive
from ribs.visualize import grid_archive_heatmap

from src.env_search import _extract_scheduler_nevals, save_heatmap
from src.libero_spatial_eval import LiberoSpatialEval
from src.measures import PCAMeasure


def show_interactive_archive(archive: GridArchive, num_trials_per_sol: int = 4):
    fig = plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=1, cmap="viridis")
    plt.tight_layout()

    def onclick(event):
        occupied, data = archive.retrieve_single([event.xdata, event.ydata])

        if occupied:
            print(
                f"Recorded objective: {data['objective']}; Recorded measures: {data['measures']}"
            )
            evaluator = LiberoSpatialEval(
                task_id=data["task_id"], num_trials_per_sol=num_trials_per_sol
            )
            _, obj, meas, _ = evaluator.evaluate_single(
                solution=data["solution"]
            )
            print(f"Evaluated objective: {obj}; Evaluated measures: {meas}")
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
    archive: GridArchive, num_trials_per_sol: int = 4, port: int = 8050
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
            evaluator = LiberoSpatialEval(
                task_id=data["task_id"], num_trials_per_sol=num_trials_per_sol
            )
            _, obj, meas, _ = evaluator.evaluate_single(
                solution=data["solution"]
            )
            print(f"Evaluated objective: {obj}; Evaluated measures: {meas}")
        else:
            print("Archive cell not occupied")

        return None

    app.run(host="0.0.0.0", port=port, debug=True)


def success_rates_on_envs(
    env_archive: ArchiveBase,
    num_trials_per_sol: int = 4,
    heatmap_savepath="success_rates.png",
    archive_savepath="success_rates.pkl",
):
    evaluators = [
        LiberoSpatialEval(
            task_id=tid,
            objective_func="success_rate",
            measure_func="spread_similarity",
            num_trials_per_sol=num_trials_per_sol,
        )
        for tid in range(10)
    ]

    dummy_archive = GridArchive(
        solution_dim=env_archive.solution_dim,
        dims=2,
        ranges=[[0, 1], [0, 1]],
    )

    for cell in tqdm.tqdm(env_archive):
        _, objective, measures, _ = evaluators[cell["task_id"]].evaluate_single(
            solution=cell["solution"]
        )

        dummy_archive.add_single(
            solution=cell["solution"], objective=objective, measures=measures
        )

        with open(archive_savepath, "wb") as f:
            pkl.dump(dummy_archive, f)

    save_heatmap(dummy_archive, heatmap_savepath)


def redraw_heatmap(
    experiment_logdir: str,
    encoder_ckpt_path: str,
):
    measure_model = PCAMeasure(ckpt_path=encoder_ckpt_path)

    all_nevals = _extract_scheduler_nevals(experiment_logdir)

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

        save_heatmap(
            dummy_archive, f"{experiment_logdir}/heatmap_{nevals:08d}.png"
        )


if __name__ == "__main__":
    # with open(
    #     # Enter the scheduler checkpoint you want to visualize here
    #     file="scheduler_00000500.pkl",
    #     mode="rb",
    # ) as f:
    #     archive = pkl.load(f).archive
    #     # show_interactive_archive(archive, num_trials_per_sol=4)
    #     host_interactive_archive(archive, num_trials_per_sol=4)
    #     # success_rates_on_envs(archive, num_trials_per_sol=4)

    from sklearn.metrics.pairwise import rbf_kernel
    from vendi_score import vendi

    all_dists = []
    all_nevals = []
    for filename, nevals in _extract_scheduler_nevals(
        "domain_randomization/"
    ).items():
        print(filename)
        all_nevals.append(nevals)
        with open(file=filename, mode="rb") as f:
            archive_data = pkl.load(f).result_archive.data(
                ["objective", "embedding"]
            )
            rbf_K = rbf_kernel(X=archive_data["embedding"], gamma=0.002)
            emb_vendi_rbf = vendi.score_K(rbf_K, normalize=True)
            qvendi = np.mean(archive_data["objective"]) * emb_vendi_rbf
            all_dists.append(qvendi)
            print(qvendi)
    all_dists = np.array(all_dists)
    all_dists = all_dists[np.argsort(all_nevals)]
    print(all_dists)

    # from pathlib import Path

    # from src.dataset_utils import TempDataset

    # temp_succ_dataset = TempDataset(
    #     dataset_dir=Path("domain_randomization/succ_dataset/")
    # )
    # temp_succ_dataset.convert_to_lerobot(
    #     "domain_randomization", max_traj_len=100
    # )
