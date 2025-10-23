import json
import pickle as pkl
from itertools import product

import torch
from ribs.archives import ArchiveBase
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from qd_spatial import Trajectory


class LazyPreferenceDataset(Dataset):
    def __init__(self, pair_filename: str, archive: ArchiveBase) -> None:
        """This dataset object loads training pairs directly from the QD 
        archive according to archive and trajectory indices specified in 
        a separate JSONL file.

        Args:
            pair_filename (str): Path to a jsonl file specifying pairs of 
                comparable success and fail trajectories. 
                
                Example:
                    [
                        {
                            "success": (
                                <archive_index1>, # ArchiveBase.index_of(measures)
                                <trajectory_index1>
                            ), 
                            "fail" (
                                <archive_index1>, # same as success
                                <trajectory_index2>
                            )
                        }
                        ...
                    ]

            archive (ArchiveBase): The QD archive which contains rollout 
                trajectories for each task. All rollouts from the same archive 
                cell are run on the same task using the same model, so pairs of 
                trajectories from the same cell should be comparable to each 
                other.
        """
        with open(pair_filename) as pair_file:
            self.pairs: list[dict[str, tuple[int, int]]] = [json.loads(line) for line in pair_file]
        
        self.archive: ArchiveBase = archive

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Trajectory]:
        success_idxs = self.pairs[idx]['success']
        fail_idxs = self.pairs[idx]['fail']

        assert success_idxs[0] == fail_idxs[0], \
            f"Success traj index {success_idxs[0]} and fail traj index {fail_idxs[0]} do not match!"
        
        # first index finds archive cell         
        all_trajectories = self.archive._store.retrieve(success_idxs[0])['trajectories'] # type: ignore
        
        return {
            # second index finds trajectory
            "success": all_trajectories[success_idxs[1]], 
            "fail": all_trajectories[fail_idxs[1]]
        }
    

def generate_pair_file(archive: ArchiveBase, save_path: str) -> None:
    """Assembles preference pairs each consisted of a success and fail rollout 
    trajectory, and saves all pairs to a JSONL file.

    Args:
        archive (ArchiveBase): The QD archive which contains rollout 
            trajectories for each task. All rollouts from the same archive 
            cell are run on the same task using the same model, so pairs should 
            be formed between trajectories from the same cell.
        save_path (str): Where to save the JSONL file containing all preference 
            pairs.
    """
    pairs: list[dict[str, tuple[int, int]]] = []
    for cell in tqdm(archive):
        all_trajectories: list[Trajectory] = cell["trajectories"]

        success_traj_idxs: list[int] = []
        fail_trajs_idxs: list[int] = []
        for idx, traj in enumerate(all_trajectories):
            if traj.success:
                success_traj_idxs.append(idx)
            else:
                fail_trajs_idxs.append(idx)

        # All success or all fail, no pair
        if len(success_traj_idxs) == 0 or len(fail_trajs_idxs) == 0:
            continue
        
        # Add all pairs of success and fail trajectories from the same cell
        # Save both archive cell idx and traectory idx for later access
        archive_idx = archive.index_of_single(cell["measures"]) # type: ignore
        for success_idx, fail_idx in product(success_traj_idxs, fail_trajs_idxs):
            pairs.append({
                "success": (archive_idx, success_idx),
                "fail": (archive_idx, fail_idx)
            })

        with open(save_path, "w") as pair_file:
            for p in pairs:
                pair_file.write(json.dumps(p) + "\n")

if __name__ == '__main__':
    with open(
        file="scheduler_00001000.pkl",
        mode="rb",
    ) as f:
        archive = pkl.load(f).result_archive
        loader = DataLoader(
            LazyPreferenceDataset("pairs.jsonl", archive),
            batch_size=32, 
            shuffle=True, 
            num_workers=4
        )

        for batch in loader:
            # batch["success"]
            break