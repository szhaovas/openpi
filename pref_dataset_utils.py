import json
import pickle as pkl
from itertools import permutations

from tqdm import tqdm

from typing import List, Dict
import fire
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import numpy as np
from dataclasses import dataclass, field
import contextlib
import imageio


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
        }

        dataset: LeRobotDataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=10,
            features=dataset_features,
            image_writer_threads=10,
            image_writer_processes=5,
        )

        # traj_counter = 0
        # all_pairs = []
        # progress = tqdm(archive)
        # for cell in progress:
        #     # If all trajectories in the cell failed, we see the cell's stored 
        #     # env as infeasible and do not include its trajectories into the 
        #     # finetune dataset
        #     if (
        #         cell['objective'] < 1e-3
        #         # TODO: Do we include cells in which all trajectories succeeded?
        #         # and not cell["trajectories"][0].success
        #     ):
        #         continue

        #     scores = {}
        #     for idx, traj in enumerate(cell["trajectories"]):
        #         with _suppress_tqdm():
        #             _add_trajectory_to_dataset(
        #                 traj,
        #                 dataset
        #             )
                
        #         scores[traj_counter] = 0
        #         imageio.mimwrite(f"traj_{idx}.mp4", traj.image, fps=10)
        #         traj_counter += 1

        #     progress.clear()
        #     # Review the trajectory videos under pwd and rank them in ascending order based on your preference.
        #     user_scores = []
        #     while len(user_scores) != len(scores):
        #         user_scores = [int(x) for x in input("Type in scores for ALL trajectories separated by comma: ").replace(",", " ").split()]
        #     for traj_id, score in zip(scores.keys(), user_scores):
        #         scores[traj_id] = score
        #     cell_pairs = [(cid, rid) for cid, rid in permutations(scores, 2) if scores[cid] > scores[rid]]
        #     print(f"Adding pairs: {cell_pairs}")
        #     all_pairs.extend(cell_pairs)
        #     progress.refresh()

        traj_counter = 0
        all_pairs = []
        for cell in tqdm(archive):
            # If all trajectories in the cell failed, we see the cell's stored 
            # env as infeasible and do not include its trajectories into the 
            # finetune dataset
            if (
                cell['objective'] < 1e-3
                # TODO: Do we include cells in which all trajectories succeeded?
                # and not cell["trajectories"][0].success
            ):
                continue

            scores = {}
            for traj in cell["trajectories"]:
                with _suppress_tqdm():
                    _add_trajectory_to_dataset(
                        traj,
                        dataset
                    )
                
                # big score for successful trajectories to make sure they always 
                # rank ahead of failed trajectories
                this_score = 0
                if traj.success:
                    this_score += 100
                
                # penalize by eef pose variance
                this_score -= np.sum(np.var(np.stack(traj.state)[:,:-1], axis=0))
                
                scores[traj_counter] = this_score
                
                traj_counter += 1

            cell_pairs = [(cid, rid) for cid, rid in permutations(scores, 2) if scores[cid] > scores[rid]]
            all_pairs.extend(cell_pairs)

        with open(HF_LEROBOT_HOME / repo_id / 'pairs.jsonl', "w") as pair_file:
            for cid, rid in all_pairs:
                pair_file.write(json.dumps({"chosen": cid, "rejected": rid}) + "\n")

        print(f"Saved pref dataset to {HF_LEROBOT_HOME / repo_id}")

if __name__ == '__main__':
    # python pref_dataset_utils.py --scheduler_path=test_logs/scheduler_00000020.pkl
    fire.Fire(generate_all_train_assets)