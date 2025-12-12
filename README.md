## Installation
```bash
git clone --recurse-submodules -b pref_learning https://github.com/szhaovas/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
```

## Training
<!-- (FIXME: Need to figure out how to convert our dataset to v2 to allow automatic download) -->
Download [our dataset](https://huggingface.co/datasets/physical-intelligence/libero) and move it under `~/.cache/huggingface/lerobot/`. Create parent directories if they don't exist.
```python
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/train.py pi0_fast_libero_pref --exp-name=my_experiment --overwrite
```

## Visualization
<!-- (FIXME: Might have copyright issues) -->
Download [CPLEX_Studio](https://drive.google.com/file/d/1Tktk-vV-HvyuSWAmTTHVikATFeqv5yL7/view?usp=sharing) and unzip to `third_party/CPLEX_Studio201`. Then install it and some other dependencies to the LIBERO venv:
```bash
source examples/libero/.venv/bin/activate
uv run third_party/CPLEX_Studio201/python/setup.py install
uv pip install dash
```
Within the LIBERO venv, run:
```python
python viz_spatial_attack.py
```
Open a second terminal and run:
```python
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint --policy.config pi0_fast_libero_pref --policy.dir <your_finetuned_checkpoint>
```
This will display an interactive archive heatmap at `localhost:8050`. You can view it in the browser and click on a cell to save rollouts of that cell's solution to `interactive_vids`. If you are on ssh, you can also configure port forwarding to view and interact with heatmap on your own computer.