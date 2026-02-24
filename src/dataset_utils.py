import contextlib
import glob
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import imageio
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

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


@dataclass
class Trajectory:
    prompt: Optional[str] = (
        None  # TODO: Find some way to merge this with env params
    )
    success: bool = False
    image: List[np.ndarray] = field(default_factory=list)
    wrist_image: List[np.ndarray] = field(default_factory=list)
    state: List[np.ndarray] = field(default_factory=list)
    action: List[np.ndarray] = field(default_factory=list)
    embedding: List[np.ndarray] = field(default_factory=list)


class TempDataset(Dataset):
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir

        if self.dataset_dir.is_dir():
            self._num_eps = len(glob.glob(f"{self.dataset_dir}/ep_{'[0-9]'*5}"))
        else:
            dataset_dir.mkdir(parents=True)
            self._num_eps = 0

    def __getitem__(self, idx: int) -> Trajectory:
        eps_dir = self.dataset_dir / f"ep_{idx:05d}"

        state_action_embedding = np.load(eps_dir / "state_action_embedding.npz")
        state = state_action_embedding["state"]
        action = state_action_embedding["action"]
        embedding = state_action_embedding["embedding"]

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
            action=action,
            embedding=embedding,
        )

    def __len__(self) -> int:
        return self._num_eps

    def __iter__(self) -> Iterator[Trajectory]:
        for ep_idx in range(self._num_eps):
            yield self[ep_idx]

    def write_episode(self, trajectory: Trajectory, fps: int = 10) -> None:
        if trajectory.prompt is None:
            logger.warning("Cannot write empty trajectory")
            return
        
        traj_id = len(self)
        eps_dir = self.dataset_dir / f"ep_{traj_id:05d}"
        eps_dir.mkdir()

        np.savez(
            eps_dir / "state_action_embedding.npz",
            state=trajectory.state,
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

        self._num_eps += 1

        return traj_id

    def convert_to_lerobot(
        self, repo_id: str, max_traj_len: Optional[int] = None
    ) -> LeRobotDataset:
        """Converts this dataset to LeRobot format suitable for openpi finetuning

        Args:
            repo_id: The repo ID that determines where the output dataset
                will be saved. By default, the save location will be
                ``HF_LEROBOT_HOME / repo_id``
            max_traj_len: Only trajectories shorter than ``max_traj_len`` will
                get exported. This is for filtering out low-quality successful
                trajectories.


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
            if max_traj_len is None or len(trajectory.state) < max_traj_len:
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

        print(f"Saved dataset to {HF_LEROBOT_HOME / repo_id}")

        return dataset

    def convert_to_rlds(self):
        raise NotImplementedError


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
