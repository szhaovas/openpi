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
                "task": trajectory.prompt
            }
        )

    dataset.save_episode()
    

def generate_all_train_assets(
    scheduler_path: str = "test_logs/scheduler_00001000", 
    success_repo_id: str = "pref/success", 
    fail_repo_id: str = "pref/fail", 
    pair_savepath: str = "pairs.jsonl"
) -> None:
    """Assembles preference pairs each consisted of a success and fail rollout 
    trajectory, and saves all pairs to a JSONL file.
    """
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
            }
        }

        success_dataset: LeRobotDataset = LeRobotDataset.create(
            repo_id=success_repo_id,
            robot_type="panda",
            fps=10,
            features=dataset_features,
            image_writer_threads=10,
            image_writer_processes=5,
        )
        fail_dataset: LeRobotDataset = LeRobotDataset.create(
            repo_id=fail_repo_id,
            robot_type="panda",
            fps=10,
            features=dataset_features,
            image_writer_threads=10,
            image_writer_processes=5,
        )

        success_episode_counter = 0
        fail_episode_counter = 0
        pairs: List[Dict[str, int]] = []
        for cell in tqdm(archive): 
            success_traj_idx: List[int] = []
            fail_traj_idx: List[int] = []
            for traj in cell["trajectories"]:
                if traj.success:
                    with _suppress_tqdm():
                        _add_trajectory_to_dataset(
                            traj,
                            success_dataset
                        )
                    success_traj_idx.append(success_episode_counter)
                    success_episode_counter += 1
                else:
                    with _suppress_tqdm():
                        _add_trajectory_to_dataset(
                            traj,
                            fail_dataset
                        )
                    fail_traj_idx.append(fail_episode_counter)
                    fail_episode_counter += 1

            # All success or all fail, no pair
            if len(success_traj_idx) == 0 or len(fail_traj_idx) == 0:
                continue
            
            # Add all pairs of success and fail trajectories from the same cell
            # Save both archive cell idx and traectory idx for later access
            for success_idx, fail_idx in product(success_traj_idx, fail_traj_idx):
                pairs.append({"success": success_idx, "fail": fail_idx})

        with open(pair_savepath, "w") as pair_file:
            for p in pairs:
                pair_file.write(json.dumps(p) + "\n")

        print(f"Updated success rollout dataset at {HF_LEROBOT_HOME / success_repo_id}")
        print(f"Updated fail rollout dataset at {HF_LEROBOT_HOME / fail_repo_id}")

if __name__ == '__main__':
    # python pref_dataset_utils.py --scheduler_path=test_logs/scheduler_00000020.pkl
    fire.Fire(generate_all_train_assets)