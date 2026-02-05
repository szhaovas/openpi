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
./run_env_search.sh cma_mae # cma_mae or domain_randomization
```
Some additional fields at the top of `run_env_search.sh` that can be changed:
- `NUM_SERVERS`: The number of GPUs you wish to use to host VLA servers. We parallelize rollouts, and the number of rollouts on each generated environment is set to `NUM_SERVERS` by default. If you only have access to a single GPU, set `NUM_SERVERS=1` and modify `<config.eval.task_eval.num_trials_per_sol>` to the desired number of rollouts (in this case, rollouts will be run sequentially).
- `SERVER_PORT_START`: Defines local ports on which VLAs will communicate with the pipeline. Each VLA server is assigned its own port, so `SERVER_PORT_START, SERVER_PORT_START+1,...,SERVER_PORT_START+NUM_SERVERS-1` will be assigned. You shouldn't need to change this unless other processes are using the default ports.
- `GPU_ID_START`: Defines the CUDA device IDs on which VLAs will be hosted. `GPU_ID_START,GPU_ID_START+1,...GPU_ID_START+NUM_SERVERS-1` will be assigned. You shouldn't need to change this unless other processes are using GPU0~.

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
This will display an interactive archive heatmap at `localhost:8050`. You can view it in the browser and click on a cell to save rollouts of that cell's solution to `interactive_vids`. If you are on ssh, you can also configure port forwarding to view and interact with heatmap on your own computer.