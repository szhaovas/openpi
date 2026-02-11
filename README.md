## Installation
Please make sure you are not in a virtual environment before running this.
```bash
git clone --recurse-submodules https://github.com/szhaovas/openpi.git
cd openpi
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [tmux](https://github.com/tmux/tmux/wiki/Installing).

<!-- FIXME: Might have copyright issues with CPLEX -->
Download [CPLEX_Studio](https://drive.google.com/file/d/1Tktk-vV-HvyuSWAmTTHVikATFeqv5yL7/view?usp=sharing) and 
unzip to pwd.

```bash
# Install libero and qd dependencies
uv venv --python 3.10
uv pip sync requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

# Install CPLEX_Studio
source .venv/bin/activate
python -m ensurepip
python CPLEX_Studio201/python/setup.py install
```

## Run QD search
```bash
./run_env_search.sh <envgen> <vla>
```
- \<envgen>: The environment generation algorithm. Currently can be either `domain_randomization` or `cma_mae`.
- \<vla>: The vla with which to collect rollouts. Currently can be either `openpi` or `openvla`

Some additional fields at the top of `run_env_search.sh` that can be changed:
- `VLA_SERVER_URIs`: The IPs and ports on which to host VLA servers. The number 
 of rollouts is set to the number of URIs by default since we parallelize by 
 rollouts. If you only have access to a single GPU, set this as a single URI 
 (e.g. `0.0.0.0:8000`) and modify `<config.eval.task_eval.num_trials_per_sol>` 
 to the desired number of rollouts (in this case, rollouts will be run 
 sequentially).
- `GPU_IDs`: Defines the CUDA device IDs on which to host VLA servers. This 
should have the same length as `VLA_SERVER_URIs`.

QD search will save the finetuning dataset at `~/.cache/huggingface/lerobot/<config.envgen>`.

## Finetune
Compute normalization stats:
```bash
cd openpi
uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune --envgen_dataset_repo_id <config.envgen>
```
LoRa SFT:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/train.py pi0_fast_libero_<config.envgen> --exp-name=my_experiment --overwrite
```

## Visualization
Terminal 1:
```bash
uv run visualization.py
```
Terminal 2:
```bash
cd openpi
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint --policy.config pi0_fast_libero_low_mem_finetune --policy.dir <your_finetuned_checkpoint>
```
<!-- CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --env LIBERO --port 8001 policy:checkpoint --policy.config pi0_fast_libero_cma_mae --policy.dir checkpoints/pi0_fast_libero_envgen/cma_mae/29999 -->

This will display an interactive archive heatmap at `localhost:8050`. You can view it in the browser and click on a cell to save rollouts of that cell's solution to `interactive_vids`. If you are on ssh, you can also configure port forwarding to view and interact with heatmap on your own computer.