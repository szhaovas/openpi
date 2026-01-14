## Installation
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
./run_qd_search.sh
```
Change the `NUM_SERVERS` field in `run_qd_search.sh` to the number of GPUs you wish to use to host VLA servers. By default, this is also the number of times each generated environment will be evaluated.

## Finetune
<!-- TODO: Add instructions for computing norm stats -->
```python
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/train.py pi0_fast_libero_low_mem_finetune --exp-name=my_experiment --overwrite
```

## Visualization
Terminal 1:
```python
uv run visualization.py
```
Terminal 2:
```bash
cd openpi
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint --policy.config pi0_fast_libero_low_mem_finetune --policy.dir <your_finetuned_checkpoint>
```
This will display an interactive archive heatmap at `localhost:8050`. You can view it in the browser and click on a cell to save rollouts of that cell's solution to `interactive_vids`. If you are on ssh, you can also configure port forwarding to view and interact with heatmap on your own computer.