import json
import pickle as pkl
from itertools import product

from tqdm import tqdm

from typing import List, Dict
import fire
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import numpy as np
from dataclasses import dataclass, field
import contextlib

# FIXME: This script needs 3.10 but libero needs 3.8. Maybe move this to a third file to resolve dependency
@dataclass
class Trajectory:
    prompt: str
    success: bool = False
    image: List[np.ndarray] = field(default_factory=list)
    wrist_image: List[np.ndarray] = field(default_factory=list)
    state: List[np.ndarray] = field(default_factory=list)
    action: List[np.ndarray] = field(default_factory=list)
    # TODO: store policy embeddings(?)


@contextlib.contextmanager
def _suppress_tqdm():
    original = tqdm.__init__
    def disabled_init(self, *args, **kwargs):
        kwargs["disable"] = True
        original(self, *args, **kwargs)
    tqdm.__init__ = disabled_init
    try:
        yield
    finally:
        tqdm.__init__ = original


def _add_trajectory_to_dataset(
    trajectory: Trajectory,
    dataset: LeRobotDataset
) -> None:
    for image, wrist_image, state, action in zip(
        trajectory.image,
        trajectory.wrist_image,
        trajectory.state,
        trajectory.action
    ):
        dataset.add_frame(
            {
                "image": image,
                "wrist_image": wrist_image,
                "state": state,
                "actions": action[0], # FIXME: save only the current action
                "task": trajectory.prompt,
                "success": np.array([trajectory.success])
            }
        )

    dataset.save_episode()
    

def generate_all_train_assets(
    scheduler_path: str = "test_logs/scheduler_00001000.pkl", 
    repo_id: str = "shihanzh/qdpref"
) -> None:
    with open(file=scheduler_path, mode="rb") as archive_file:
        archive = pkl.load(archive_file).result_archive
        
        dataset_features = {
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
            "success": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["success"],
            }
        }

        dataset: LeRobotDataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=10,
            features=dataset_features,
            image_writer_threads=10,
            image_writer_processes=5,
        )

        for cell in tqdm(archive): 
            # If all trajectories in the cell failed, we see the cell's stored 
            # env as infeasible and do not include its trajectories into the 
            # finetune dataset
            if (
                cell['objective'] < 1e-5
                # TODO: Do we include cells in which all trajectories succeeded?
                and not cell["trajectories"][0].success
            ):
                continue

            for traj in cell["trajectories"]:
                with _suppress_tqdm():
                    _add_trajectory_to_dataset(
                        traj,
                        dataset
                    )

        print(f"Updated rollout dataset at {HF_LEROBOT_HOME / repo_id}")

if __name__ == '__main__':
    # python pref_dataset_utils.py --scheduler_path=test_logs/scheduler_00000020.pkl
    fire.Fire(generate_all_train_assets)