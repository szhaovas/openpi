# TODO(Shihan): I intended for this script to load the data within `rollouts`
#   and combine it into a dataset ready to be fed into LSTM VAE training. Right 
#   now it is not functional but I've copied over some codes from SAFE and 
#   removed things I think we won't need

import glob
import pickle
from dataclasses import dataclass
from pathlib import Path

import natsort
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class Rollout:
    '''
    A single rollout of the experiment.
    '''
    hidden_states: torch.Tensor
    task_suite_name: str
    task_id: int
    task_description: int
    episode_idx: int
    episode_success: int
    
    def __post_init__(self):
        self.episode_success = int(self.episode_success)

class RolloutDataset(Dataset):
    '''
    A PyTorch Dataset for the rollout data.
    '''
    def __init__(
        self, 
        rollouts: list[Rollout], 
        device="cuda"
    ):
        self.rollouts = rollouts
        self.length = len(rollouts)
        self.device = device
        
        features = pad_rollout_batch(self.rollouts, self.device)
        self.features = features
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = {
            'features': self.features[idx],
        }
        return data
    
    def get_rollouts(self):
        return self.rollouts
    
    def get_features(self):
        return self.features


def pad_rollout_batch(
    rollouts: list[Rollout], device = None
):
    """
    Pad the hidden states to the same length (max length in the batch).

    Args:
        rollouts: list of rollouts, each containing:
            - hidden_states (Tensor): shape [sequence_length, hidden_dim]
        device: device to put the tensors on

    Returns:
        padded_features (Tensor): shape [batch_size, max_length, hidden_dim]
    """
    # Extract all hidden states into a list
    batch_features = [r.hidden_states for r in rollouts]

    # Determine padding dimensions
    max_length = max(seq.shape[0] for seq in batch_features)
    hidden_dim = batch_features[0].shape[-1]
    batch_size = len(batch_features)

    # Infer dtype and device from the first sequence
    dtype = batch_features[0].dtype
    if device is None:
        device = batch_features[0].device

    # Pre-allocate output tensors
    padded_features = torch.zeros(
        batch_size, max_length, hidden_dim,
        dtype=dtype, device=device
    )

    # Fill in values for each sequence
    for i, seq in enumerate(batch_features):
        seq_length = seq.shape[0]
        padded_features[i, :seq_length] = seq.to(device)

    return padded_features

# TODO(Shihan): This is where they aggregate the policy embedding data, I haven't
# changed it, but for our purpose I think it'd be easier if we just ignore these 
# and do aggregation within pi0/pi0-fast
# def parse_and_index_tensor_last(A, command):
#     """
#     Parse a command string to index into the last two dimensions of a multi-dimensional tensor A,
#     and then flatten these last two dimensions.

#     Supported commands:
#       - "concat": 
#           -> Flatten the entire last two dimensions (of shape (c, d) becomes (c*d,)).
#       - "concat-:10" or "concat-::5": 
#           -> Apply a Python slice on the second-to-last axis (i.e. the "row" axis of the last two dims),
#              then flatten the last two dimensions.
#       - "concat-2", "concat-5", etc.: 
#           -> Uniformly index into the second-to-last dimension to obtain the specified number of features.
#              For instance, "concat-2" selects the first and last positions along that dimension.
#              "concat-5" selects 5 indices (first, last, and three equally spaced indices in between).

#     Parameters:
#       A (np.ndarray): A multi-dimensional tensor (e.g., shape (..., c, d)).
#       command (str): A command starting with "concat" that specifies how to index the tensor.

#     Returns:
#       np.ndarray: The tensor where the operation has been applied on the last two dimensions and then flattened.

#     Raises:
#       ValueError: If the command format is not recognized.
#     """
#     # Case 1: When command is exactly "concat", flatten the last two dimensions.
#     if command == "concat":
#         new_last_dim = A.shape[-2] * A.shape[-1]
#         return A.reshape(*A.shape[:-2], new_last_dim)
    
#     prefix = "concat-"
#     sub_cmd = command[len(prefix):]  # Extract portion after "concat-"
    
#     # Check if the sub-command contains a colon (indicating slice notation)
#     if ":" in sub_cmd:
#         parts = sub_cmd.split(':')
#         # Two-part slice, e.g. ":10"
#         if len(parts) == 2:
#             start_str, stop_str = parts
#             start = int(start_str) if start_str != "" else None
#             stop = int(stop_str) if stop_str != "" else None
#             # Apply slicing on the second-to-last dimension.
#             indexed = A[..., slice(start, stop), :]
#         # Three-part slice, e.g. "::5"
#         elif len(parts) == 3:
#             start_str, stop_str, step_str = parts
#             start = int(start_str) if start_str != "" else None
#             stop = int(stop_str) if stop_str != "" else None
#             step = int(step_str) if step_str != "" else None
#             indexed = A[..., slice(start, stop, step), :]
#         else:
#             raise ValueError("Invalid slice format in command.")
        
#         new_last_dim = indexed.shape[-2] * indexed.shape[-1]
#         return indexed.reshape(*indexed.shape[:-2], new_last_dim)
    
#     # Otherwise, check if the sub-command is simply an integer for uniform indexing.
#     try:
#         k = int(sub_cmd)
#     except ValueError:
#         raise ValueError("Invalid command format; expected a colon-based slice or an integer.")
    
#     # Uniform indexing requires at least 2 features.
#     if k < 2:
#         raise ValueError("Uniform indexing requires at least 2 features.")
    
#     # Determine the number of indices available along the second-to-last dimension.
#     c = A.shape[-2]
#     # Compute k indices uniformly spaced, including the endpoints.
#     indices = np.linspace(0, c - 1, num=k)
#     # Convert to integers by rounding.
#     indices = np.round(indices).astype(int)
#     # Use these indices to select along the second-to-last dimension.
#     indexed = A[..., indices, :]
#     new_last_dim = indexed.shape[-2] * indexed.shape[-1]
#     return indexed.reshape(*indexed.shape[:-2], new_last_dim)
    
    
# def process_tensor_idx_rel(A, command):
#     """
#     Process a multi-dimensional tensor A based on a provided command.

#     The command specifies the operation to perform on A as follows:
    
#       1. If `command` is a float (between 0 and 1):
#          - Interprets it as a relative index to select a single token from the second-to-last dimension.
#          - Example: token_idx = round((A.shape[-2]-1) * command) 
#            and then returns A[..., token_idx, :].

#       2. If `command` is the string "mean":
#          - Computes the mean over the second last axis 
#            (Note: the axis choice here is based on your snippet; if you intend to average over the horizon axis of the last two dimensions, you might use axis=-2 instead).
      
#       3. If `command` is a string containing "concat":
#          - Calls `parse_and_index_tensor_last(A, command)` to apply a slice to the last two dimensions and flatten them.
      
#       4. Otherwise:
#          - Raises a ValueError indicating an unknown token index.

#     Parameters:
#       A (np.ndarray): A multi-dimensional tensor.
#       command (str or float): The command specifying which processing operation to apply.
      
#     Returns:
#       np.ndarray: The processed tensor.

#     Raises:
#       ValueError: If the command is not recognized.
#     """
#     assert len(A.shape) >= 2, "Tensor A must have at least two dimensions."
    
#     if isinstance(command, float):
#         # Validate the command as a float in the range [0, 1].
#         assert 0 <= command <= 1, f"Invalid token index ratio: {command}"
#         token_idx = round((A.shape[-2] - 1) * command)
#         # Select the specific token along the second-to-last dimension.
#         return A[..., token_idx, :]
    
#     elif command == "mean":
#         # Compute the mean over axis 0.
#         # (Adjust the axis if you need the mean over a different dimension.)
#         return A.mean(axis=-2)
    
#     elif isinstance(command, str) and "concat" in command:
#         return parse_and_index_tensor_last(A, command)
    
#     else:
#         raise ValueError(f"Unknown token index: {command}")

# TODO(Shihan): Copied from failure_prob/data/pizero.py. This is for pi0-libero 
#   setting. You may need to update this.
def load_rollouts_from_root(load_root: Path) -> list[Rollout]:
    env_records_folder = load_root / "env_records"
    policy_records_folder = load_root / "policy_records"
    
    assert env_records_folder.exists(), f"Path {env_records_folder} does not exist"
    assert policy_records_folder.exists(), f"Path {policy_records_folder} does not exist"
    
    env_record_paths = glob.glob(str(env_records_folder / "*.pkl"))
    policy_record_paths = glob.glob(str(policy_records_folder / "*meta.pkl"))
    
    assert len(env_record_paths) > 0, f"No env records found in {env_records_folder}"
    assert len(policy_record_paths) > 0, f"No policy records found in {policy_records_folder}"
    
    env_record_paths = natsort.natsorted(env_record_paths)
    policy_record_paths = natsort.natsorted(policy_record_paths)
    
    all_rollouts = []
    
    policy_step = 0

    for env_record_path in tqdm(env_record_paths):
        # Load the meta data from the env record
        env_record = pickle.load(open(env_record_path, "rb"))
        
        # Load hidden features from corresponding policy records
        model_infer_times = env_record["model_infer_times"]
        policy_records = []
        for i in range(model_infer_times):
            policy_record_path = policy_record_paths[policy_step]
            policy_records.append(pickle.load(open(policy_record_path, "rb")))
            policy_step += 1
            
        # Extract hidden states and actions from policy records
        hidden_states = []
        for policy_record in policy_records:
            # hidden_state shape: (n_diff_steps, n_pred_horizon, dim_feats)
            hidden_state = policy_record['']
            
            # # handle the pred_horizon dimension
            # # (n_diff_steps, n_pred_horizon, dim_feats) -> (n_diff_steps, dim_feats)
            # hidden_state = process_tensor_idx_rel(hidden_state, cfg.dataset.horizon_idx_rel)
            
            # # handle the diff_steps dimension
            # # (n_diff_steps, dim_feats) -> (dim_feats)
            # hidden_state = process_tensor_idx_rel(hidden_state, cfg.dataset.diff_idx_rel)
            
            hidden_states.append(hidden_state)
            
        hidden_states = np.stack(hidden_states, axis=0).astype(np.float32)
        hidden_states = torch.from_numpy(hidden_states) # (n_steps, hidden_dim)
        
        rollout = Rollout(
            hidden_states=hidden_states,
            task_suite_name=env_record["task_suite_name"],
            task_id=env_record["task_id"],
            task_description=env_record["task_description"],
            episode_idx=env_record["episode_idx"],
            episode_success=env_record["episode_success"],
        )
        all_rollouts.append(rollout)
    
    return all_rollouts