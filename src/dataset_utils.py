import contextlib
import glob
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import imageio
import numpy as np
import tensorflow_datasets as tfds
import torch
from lerobot.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


lerobot_dataset_features = {
    "image": {
        "dtype": "image",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["state"],
    },
    "actions": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["actions"],
    },
}

rlds_dataset_features = tfds.features.FeaturesDict(
    {
        "steps": tfds.features.Dataset(
            {
                "observation": tfds.features.FeaturesDict(
                    {
                        "image": tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Main camera RGB observation.",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Wrist camera RGB observation.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc="Robot EEF state (6D pose, 2D gripper).",
                        ),
                        "joint_state": tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc="Robot joint angles.",
                        ),
                    }
                ),
                "action": tfds.features.Tensor(
                    shape=(7,),
                    dtype=np.float32,
                    doc="Robot EEF action.",
                ),
                "discount": tfds.features.Scalar(
                    dtype=np.float32, doc="Discount if provided, default to 1."
                ),
                "reward": tfds.features.Scalar(
                    dtype=np.float32,
                    doc="Reward if provided, 1 on final step for demos.",
                ),
                "is_first": tfds.features.Scalar(
                    dtype=np.bool_, doc="True on first step of the episode."
                ),
                "is_last": tfds.features.Scalar(
                    dtype=np.bool_, doc="True on last step of the episode."
                ),
                "is_terminal": tfds.features.Scalar(
                    dtype=np.bool_,
                    doc="True on last step of the episode if it is a terminal step, True for demos.",
                ),
                "language_instruction": tfds.features.Text(
                    doc="Language Instruction."
                ),
            }
        ),
        "episode_metadata": tfds.features.FeaturesDict(
            {
                "file_path": tfds.features.Text(
                    doc="Path to the original data file."
                ),
            }
        ),
    }
)


@dataclass
class Trajectory:
    prompt: Optional[str] = (
        None  # TODO: Find some way to merge this with env params
    )
    success: bool = False
    image: List[np.ndarray] = field(default_factory=list)
    wrist_image: List[np.ndarray] = field(default_factory=list)
    state: List[np.ndarray] = field(default_factory=list)
    proprio: List[np.ndarray] = field(default_factory=list)
    action: List[np.ndarray] = field(default_factory=list)
    embedding: List[np.ndarray] = field(default_factory=list)


class TempDataset(Dataset):
    def __init__(self, dataset_dir: Path) -> None:
        if not dataset_dir.is_dir():
            dataset_dir.mkdir(parents=True)

        self.dataset_dir = dataset_dir

    def __getitem__(self, idx: int) -> Trajectory:
        eps_dir = self.dataset_dir / f"ep_{idx:05d}"

        state_proprio_action_embedding = np.load(
            eps_dir / "state_proprio_action_embedding.npz"
        )
        state = state_proprio_action_embedding["state"]
        proprio = state_proprio_action_embedding["proprio"]
        action = state_proprio_action_embedding["action"]
        if "embedding" in state_proprio_action_embedding:
            embedding = state_proprio_action_embedding["embedding"]
        else:
            embedding = []

        with open(eps_dir / "success.txt", "r", encoding="utf-8") as f:
            success = bool(f.read().rstrip("\n"))

        with open(eps_dir / "instruction.txt", "r", encoding="utf-8") as f:
            prompt = f.read().rstrip("\n")

        image = []
        with imageio.get_reader(eps_dir / "image.mp4", format="mp4") as reader:
            for frame in reader.iter_data():
                image.append(frame)

        wrist_image = []
        with imageio.get_reader(
            eps_dir / "wrist_image.mp4", format="mp4"
        ) as reader:
            for frame in reader.iter_data():
                wrist_image.append(frame)

        return Trajectory(
            prompt=prompt,
            success=success,
            image=image,
            wrist_image=wrist_image,
            state=state,
            proprio=proprio,
            action=action,
            embedding=embedding,
        )

    def __len__(self) -> int:
        return len(glob.glob(f"{self.dataset_dir}/ep_{'[0-9]'*5}"))

    def __iter__(self) -> Iterator[Trajectory]:
        for ep_idx in range(len(self)):
            yield self[ep_idx]

    def write_episode(self, trajectory: Trajectory, fps: int = 10) -> None:
        if trajectory.prompt is None:
            logger.warning("Cannot write empty trajectory")
            return

        traj_id = len(self)
        eps_dir = self.dataset_dir / f"ep_{traj_id:05d}"
        eps_dir.mkdir()

        np.savez(
            eps_dir / "state_proprio_action_embedding.npz",
            state=trajectory.state,
            proprio=trajectory.proprio,
            action=trajectory.action,
            embedding=trajectory.embedding,
        )

        with open(eps_dir / "success.txt", "w", encoding="utf-8") as f:
            f.write(f"{int(trajectory.success)}")

        with open(eps_dir / "instruction.txt", "w", encoding="utf-8") as f:
            f.write(trajectory.prompt)

        with imageio.get_writer(
            eps_dir / "image.mp4", fps=fps, codec="libx264"
        ) as writer:
            for img in trajectory.image:
                writer.append_data(img)

        with imageio.get_writer(
            eps_dir / "wrist_image.mp4", fps=fps, codec="libx264"
        ) as writer:
            for img in trajectory.wrist_image:
                writer.append_data(img)

        return traj_id

    def convert_to_lerobot(self, repo_id: str) -> LeRobotDataset:
        """Converts this dataset to LeRobot format suitable for openpi finetuning

        Args:
            repo_id: The repo ID that determines where the output dataset
                will be saved. By default, the save location will be
                ``HF_LEROBOT_HOME / repo_id``


        Returns:
            dataset (LeRobotDataset): The output dataset in LeRobot format.
                Contains features:
                    {
                        "image"
                        "wrist_image"
                        "state"
                        "actions"
                    }
        """
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=10,
            features=lerobot_dataset_features,
            image_writer_threads=10,
            image_writer_processes=5,
        )

        for trajectory in tqdm(self):
            with _suppress_tqdm():
                for image, wrist_image, state, action in zip(
                    trajectory.image,
                    trajectory.wrist_image,
                    trajectory.state,
                    trajectory.action,
                ):
                    dataset.add_frame(
                        frame={
                            "image": image,
                            "wrist_image": wrist_image,
                            "state": state.astype(
                                lerobot_dataset_features["state"]["dtype"]
                            ),
                            "actions": action.astype(
                                lerobot_dataset_features["state"]["dtype"]
                            ),
                        },
                        task=trajectory.prompt,
                    )

                dataset.save_episode()

        logger.info(f"Saved dataset to {HF_LEROBOT_HOME / repo_id}")

        return dataset

    def convert_to_rlds(self, repo_id: str):
        data_dir = Path(f"tensorflow_datasets/{repo_id}")
        builder = LiberoRLDSBuilder(traj_dataset=self, data_dir=data_dir)
        if Path(builder.data_dir).exists():
            logger.warning(f"Overwriting previous {builder.data_dir}...")
            shutil.rmtree(builder.data_dir)

        builder.download_and_prepare()

        logger.info(f"Saved dataset to {builder.data_dir}")

        return builder.as_dataset(split="train")


class LiberoRLDSBuilder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, traj_dataset: TempDataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traj_dataset = traj_dataset

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self, features=rlds_dataset_features
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(),
        }

    def _generate_examples(self):
        for traj_id, traj in enumerate(tqdm(self.traj_dataset)):
            steps = []
            for step_id in range(len(traj.action)):
                is_last = step_id == (len(traj.action) - 1)
                steps.append(
                    {
                        "observation": {
                            "image": traj.image[
                                step_id
                            ],  # No need to flip here since we already flipped when collecting from libero
                            "wrist_image": traj.wrist_image[
                                step_id
                            ],  # No need to flip here since we already flipped when collecting from libero
                            "state": traj.state[step_id].astype(np.float32),
                            "proprio": traj.proprio[step_id].astype(np.float32),
                        },
                        "action": traj.action[step_id].astype(np.float32),
                        "discount": 1.0,
                        "reward": float(
                            is_last
                        ),  # Assumes all trajectories in dataset are successful
                        "is_first": step_id == 0,
                        "is_last": is_last,
                        "is_terminal": is_last,
                        "language_instruction": traj.prompt,
                    }
                )
            yield traj_id, {
                "steps": steps,
                "episode_metadata": {
                    "file_path": "",  # Dummy, not actually used
                },
            }


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


def embedding_collate(batch: Iterable[Trajectory]) -> torch.Tensor:
    embeddings = []
    for traj in batch:
        if len(traj.embedding) > 0:
            embeddings.append(
                torch.tensor(traj.embedding[0])
            )  # Takes 1st embedding

    return torch.stack(embeddings, dim=0)


def padded_embedding_collate(
    batch: Iterable[Trajectory],
) -> Dict[str, torch.Tensor]:
    """
    This is for LSTMVAE training. Extracts the embedding from each trajectory
    in the batch and pads all embeddings up to the same length. Also saves the
    pre-pad sequence lengths for masking out reconstruction losses on padded
    steps.
    """
    embeddings, pre_pad_lengths = [], []
    for traj in batch:
        embeddings.append(torch.tensor(traj.embedding))
        pre_pad_lengths.append(len(traj.embedding))

    padded_embeddings = pad_sequence(sequences=embeddings, batch_first=True)

    return {
        "padded_embeddings": padded_embeddings,
        "pre_pad_lengths": torch.tensor(pre_pad_lengths),
    }


def filter_lerobot_dataset_by_task(
    repo_id: str,
    task_prompts: Iterable[str],
) -> LeRobotDataset:
    libero_metadata = LeRobotDatasetMetadata(repo_id)
    filtered_eps_idx = [
        eps_meta["episode_index"]
        for eps_meta in libero_metadata.episodes.values()
        if eps_meta["tasks"][0] in task_prompts
    ]

    return LeRobotDataset(repo_id, episodes=filtered_eps_idx)
