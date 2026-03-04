#!/usr/bin/env bash

ALGO=$1 # cma_mae or domain_randomization
VLA_TYPE=$2 # openpi or openvla
VLA_SERVER_URIs=(
  "0.0.0.0:8000" # space after each uri string
  "0.0.0.0:8001" # this script assumes ip to be local host
  "0.0.0.0:8002" 
  "0.0.0.0:8003" 
) # By default, the number of uris sets cfg.eval.task_eval.num_trials_per_sol
GPU_IDs=(0 0 1 1) # should have the same length as VLA_SERVER_URIs

# Check if session exists
session_name="${ALGO}-${VLA_TYPE}"
if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "Session $session_name already exists. Attaching..."
    tmux attach-session -t "$session_name"
    exit 0
fi

# Create new session
tmux new-session -d -s "$session_name"

# Spawns a pane for running QD loop
case "$VLA_TYPE" in
    openpi)
        tmux send-keys -t $session_name "
        export VLA_SERVER_URIs="$(IFS=,; echo "${VLA_SERVER_URIs[*]}")"
        uv run -m src.env_search envgen=$ALGO eval.measure_model.model_cfg.input_dim=2048
        " C-m
        ;;
    openvla)
        tmux send-keys -t $session_name "
        export VLA_SERVER_URIs="$(IFS=,; echo "${VLA_SERVER_URIs[*]}")"
        uv run -m src.env_search envgen=$ALGO eval.measure_model.model_cfg.input_dim=4096
        " C-m
        ;;
    *)
        echo "Unknown VLA: $VLA_TYPE" >&2
        tmux kill-session -t "$session_name"
        exit 1
        ;;
esac

num_servers=${#VLA_SERVER_URIs[@]}
# Spawns num_servers panes each calling serve_policy.py with its own GPU
for server_id in $(seq 0 $((num_servers-1)))
do
  if ((server_id==0)); then
    tmux split-window -h -t "$session_name"
    tmux select-pane -t "$session_name:.1"
  else
    tmux split-window -v
  fi
  case "$VLA_TYPE" in
      openpi)
          tmux send-keys -t "$session_name:0.$((server_id+1))" "
            cd openpi
            XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="${GPU_IDs[$server_id]}" uv run scripts/serve_policy.py --env LIBERO --port "${VLA_SERVER_URIs[$server_id]##*:}"
          " C-m
          ;;
      openvla)
          tmux send-keys -t "$session_name:0.$((server_id+1))" "
            cd openvla_oft
            XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="${GPU_IDs[$server_id]}" uv run -m vla_scripts.ws_vla_server --port "${VLA_SERVER_URIs[$server_id]##*:}"
          " C-m
          ;;
      *)
          echo "Unknown VLA: $VLA_TYPE" >&2
          tmux kill-session -t "$session_name"
          exit 1
          ;;
  esac
done

# Make the server panes have equal height
RIGHT_HEIGHT=$(tmux display -p -t "$session_name:.0" '#{pane_height}')
RIGHT_PANE_HEIGHT=$((RIGHT_HEIGHT / num_servers))
for server_id in $(seq 1 $num_servers)
do
  tmux resize-pane -t "$session_name:0.$server_id" -y "$RIGHT_PANE_HEIGHT"
done

# Attach
tmux attach-session -t "$session_name"