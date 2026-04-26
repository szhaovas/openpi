"""Manages storage and retrieval of rollout trajectories. Also handles exportation
to LeRobot and RLDS formats."""

import glob
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, no_type_check

import imageio
import numpy as np
import tensorflow_datasets as tfds
import torch
from lerobot.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from PIL import Image
from tensorflow_datasets.core.utils.file_utils import get_default_data_dir
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from src.easy_utils import suppress_tqdm

logger = logging.getLogger(__name__)


lerobot_dataset_features = {
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

        try:
            state_proprio_action_embedding = np.load(
                eps_dir / "state_proprio_action_embedding.npz"
            )
            state = state_proprio_action_embedding["state"]
            proprio = state_proprio_action_embedding["proprio"]
            action = state_proprio_action_embedding["action"]
        except FileNotFoundError:
            # TODO: This is for trajectories collected without proprio; remove
            # when publishing
            state_proprio_action_embedding = np.load(
                eps_dir / "state_action_embedding.npz"
            )
            state = state_proprio_action_embedding["state"]
            proprio = np.random.rand(len(state), 7)
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

    def write_episode(self, trajectory: Trajectory, fps: int = 10) -> Optional[int]:
        if trajectory.prompt is None:
            logger.warning("Cannot write empty trajectory")
            return

        traj_id = len(self)
        eps_dir = self.dataset_dir / f"ep_{traj_id:05d}"
        eps_dir.mkdir()

        np.savez(
            (
                eps_dir / "state_proprio_action_embedding.npz"
                if len(trajectory.proprio) > 0
                else eps_dir / "state_action_embedding.npz"
            ),
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
            with suppress_tqdm():
                for image, wrist_image, state, action in zip(
                    trajectory.image,
                    trajectory.wrist_image,
                    trajectory.state,
                    trajectory.action,
                ):
                    dataset.add_frame(
                        frame={
                            "image": resize_with_pad(
                                image,
                                height=lerobot_dataset_features["image"][
                                    "shape"
                                ][0],
                                width=lerobot_dataset_features["image"][
                                    "shape"
                                ][1],
                            ),
                            "wrist_image": resize_with_pad(
                                wrist_image,
                                height=lerobot_dataset_features["wrist_image"][
                                    "shape"
                                ][0],
                                width=lerobot_dataset_features["wrist_image"][
                                    "shape"
                                ][1],
                            ),
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
        data_dir = Path(get_default_data_dir()) / repo_id
        builder = LiberoSpatialNoNoops(traj_dataset=self, data_dir=data_dir)
        if Path(builder.data_dir).exists():
            logger.warning(f"Overwriting previous {builder.data_dir}...")
            shutil.rmtree(builder.data_dir)

        builder.download_and_prepare()

        logger.info(f"Saved dataset to {builder.data_dir}")

        return builder.as_dataset(split="train")


class LiberoSpatialNoNoops(tfds.core.GeneratorBasedBuilder):
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

    @no_type_check  # suppress pylance false positives on rlds_dataset_features.__getitem__
    def _generate_examples(self):
        for traj_id, trajectory in enumerate(tqdm(self.traj_dataset)):
            num_noops = 0
            steps = []
            for step_id, (
                image,
                wrist_image,
                state,
                proprio,
                action,
            ) in enumerate(
                zip(
                    trajectory.image,
                    trajectory.wrist_image,
                    trajectory.state,
                    trajectory.proprio,
                    trajectory.action,
                )
            ):
                # filter out no-ops
                prev_action = steps[-1]["action"] if len(steps) > 0 else None
                if is_noop(action, prev_action):
                    num_noops += 1
                    continue

                is_last = step_id == (len(trajectory.action) - 1)

                steps.append(
                    {
                        "observation": {
                            "image": resize_with_pad(
                                image,
                                height=rlds_dataset_features["steps"][
                                    "observation"
                                ]["image"].shape[0],
                                width=rlds_dataset_features["steps"][
                                    "observation"
                                ]["image"].shape[1],
                            ),  # No need to flip here since we already flipped when collecting from libero
                            "wrist_image": resize_with_pad(
                                wrist_image,
                                height=rlds_dataset_features["steps"][
                                    "observation"
                                ]["wrist_image"].shape[0],
                                width=rlds_dataset_features["steps"][
                                    "observation"
                                ]["wrist_image"].shape[1],
                            ),  # No need to flip here since we already flipped when collecting from libero
                            "state": state.astype(np.float32),
                            "joint_state": proprio.astype(np.float32),
                        },
                        "action": action.astype(np.float32),
                        "discount": 1.0,
                        "reward": float(
                            is_last
                        ),  # Assumes all trajectories in dataset are successful
                        "is_first": step_id == 0,
                        "is_last": is_last,
                        "is_terminal": is_last,
                        "language_instruction": trajectory.prompt,
                    }
                )

            if num_noops > 0:
                logger.info(
                    f"Filtered out {num_noops} steps with near-zero actions"
                )

            # TODO: OpenVLA-OFT training pipeline crashes on very short
            # trajectories. For now we skip them since there are very few such
            # trajectories.
            if len(steps) < 10:
                logger.warning(f"Skipped traj {traj_id} since it is too short")
                continue

            yield traj_id, {
                "steps": steps,
                "episode_metadata": {
                    "file_path": "",  # Dummy, not actually used
                },
            }


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


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return (
        np.linalg.norm(action[:-1]) < threshold
        and gripper_action == prev_gripper_action
    )


def resize_with_pad(
    images: np.ndarray, height: int, width: int, method=Image.BILINEAR
) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            _resize_with_pad_pil(
                Image.fromarray(im), height, width, method=method
            )
            for im in images
        ]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(
    image: Image.Image, height: int, width: int, method: int
) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return (
            image  # No need to resize if the image is already the correct size.
        )

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize(
        (resized_width, resized_height), resample=method
    )

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image
