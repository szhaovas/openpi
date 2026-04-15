#!/usr/bin/env bash
# Alternate version of run_env_search that allows user to start each VLA server 
# manually; useful when GPUs used by each VLA server aren't on the same cluster

ALGO=$1 # cma_mae / domain_randomization / cma_es
VLA_TYPE=$2 # pi0_fast / pi05 / openvla_oft
VLA_SERVER_URIs=(
  # "0.0.0.0:8003" # space after each uri string
  # "10.136.109.136:8003" # ip doesn't have to be local host
  "10.136.109.136:52800" # unicron 8000
  # "10.136.109.136:52801" # unicron 8001
  "10.136.109.136:51800" # primus 8000
  "10.136.109.136:51801" # primus 8001
  "10.136.109.136:50800" # momo 8000
  # "10.136.109.136:53800" # atlas 8000
) # By default, the number of uris sets cfg.eval.task_eval.num_trials_per_sol

# Check if session exists
session_name="${ALGO}-${VLA_TYPE}"
if tmux has-session -t "=$session_name" 2>/dev/null; then
    echo "Session $session_name already exists. Attaching..."
    tmux attach-session -t "$session_name"
    exit 0
fi

# Create new session
tmux new-session -d -s "$session_name"

# Spawns a pane for running QD loop
case "$VLA_TYPE" in
    pi0_fast)
        tmux send-keys -t $session_name "
        export VLA_SERVER_URIs="$(IFS=,; echo "${VLA_SERVER_URIs[*]}")"
        export VLA_TYPE="$VLA_TYPE"
        uv run -m src.env_search envgen=$ALGO eval.measure_model.model_cfg.input_dim=2048
        " C-m
        ;;
    pi05)
        tmux send-keys -t $session_name "
        export VLA_SERVER_URIs="$(IFS=,; echo "${VLA_SERVER_URIs[*]}")"
        export VLA_TYPE="$VLA_TYPE"
        uv run -m src.env_search envgen=$ALGO eval.measure_model.model_cfg.input_dim=1024
        " C-m
        ;;
    openvla_oft)
        tmux send-keys -t $session_name "
        export VLA_SERVER_URIs="$(IFS=,; echo "${VLA_SERVER_URIs[*]}")"
        export VLA_TYPE="$VLA_TYPE"
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
# Spawns a pane for each VLA server; user needs to go to each pane to launch 
# the server manually. The main process (QD) will wait for all VLA servers to 
# be launched
for server_id in $(seq 1 $num_servers)
do
  if ((server_id==1)); then
    tmux split-window -h -t "$session_name"
    tmux select-pane -t "$session_name:.1"
  else
    tmux split-window -v
  fi
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