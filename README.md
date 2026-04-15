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
- \<envgen>: The environment generation algorithm. Currently can be `domain_randomization` / `cma_mae` / `cma_es`.
- \<vla>: The vla with which to collect rollouts. Currently can be `pi0_fast` / `pi05` / `openvla_oft`

Some additional fields at the top of `run_env_search.sh` that can be changed:
- `VLA_SERVER_URIs`: The IPs and ports on which to host VLA servers. The number 
 of rollouts is set to the number of URIs by default since we parallelize by 
 rollouts. If you only have access to a single GPU, set this as a single URI 
 (e.g. `0.0.0.0:8000`) and modify `<config.eval.task_eval.num_trials_per_sol>` 
 to the desired number of rollouts (in this case, rollouts will be run 
 sequentially).
- `GPU_IDs`: Defines the CUDA device IDs on which to host VLA servers. This 
should have the same length as `VLA_SERVER_URIs`.

QD search will save the finetuning dataset at `~/.cache/huggingface/lerobot/<envgen>`.

## Finetune (OpenPi)
Compute normalization stats:
```bash
cd openpi
uv run scripts/compute_norm_stats.py --config-name <vla>_libero_<envgen>
```
LoRa SFT:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/train.py <vla>_libero_<envgen> --exp-name=my_experiment --overwrite
```

## Finetune (OpenVLA)
LoRa SFT:
```bash
cd openvla_oft
source .venv/bin/activate
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla_scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ~/tensorflow_datasets/<envgen>/libero_spatial_no_noops/1.0.0 \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir checkpoints \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 20000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note <envgen> \
  --grad_accumulation_steps 4
```

## Visualization
Terminal 1:
```bash
uv run visualization.py
```
Terminal 2:
```bash
cd openpi
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --env LIBERO --port 8000 policy:checkpoint --policy.config <vla>_libero_<envgen> --policy.dir <your_finetuned_checkpoint>
```
<!-- CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --env LIBERO --port 8001 policy:checkpoint --policy.config pi0_fast_libero_cma_mae --policy.dir checkpoints/pi0_fast_libero_envgen/cma_mae/29999 -->

This will display an interactive archive heatmap at `localhost:8050`. You can view it in the browser and click on a cell to save rollouts of that cell's solution to `interactive_vids`. If you are on ssh, you can also configure port forwarding to view and interact with heatmap on your own computer.