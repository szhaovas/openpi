import contextlib
import logging
from pathlib import Path
from typing import Iterator

import imageio
import numpy as np
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm

from src.libero_spatial_eval import Trajectory

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


class TempDataset:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir

        if self.dataset_dir.is_dir():
            self._num_eps = sum(1 for _ in self.dataset_dir.iterdir())
        else:
            dataset_dir.mkdir(parents=True)
            self._num_eps = 0

    def __getitem__(self, idx: int) -> Trajectory:
        eps_dir = self.dataset_dir / f"ep_{idx:08d}"

        state_action = np.load(eps_dir / "state_action.npz")
        state = state_action["state"]
        action = state_action["action"]

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

        eps_dir = self.dataset_dir / f"ep_{len(self):08d}"
        eps_dir.mkdir()

        np.savez(
            eps_dir / "state_action.npz",
            state=trajectory.state,
            action=trajectory.action,
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

    def convert_to_lerobot(self, repo_id: str) -> LeRobotDataset:
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
                            "state": state,
                            "actions": action,
                        },
                        task=trajectory.prompt,
                    )

                dataset.save_episode()

        print(f"Saved dataset to {HF_LEROBOT_HOME / repo_id}")

        return dataset
