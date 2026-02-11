#!/usr/bin/env bash
# Alternate version of run_env_search that allows user to start each VLA server 
# manually; useful when GPUs used by each VLA server aren't on the same cluster

EXP_NAME=$1 # cma_mae or domain_randomization
VLA_SERVER_URIs=(
  "10.136.109.136:8003" # space after each uri string
  "0.0.0.0:8003" # ip doesn't have to be local host
) # By default, the number of uris sets cfg.eval.task_eval.num_trials_per_sol

# Check if session exists
if tmux has-session -t "$EXP_NAME" 2>/dev/null; then
    echo "Session $EXP_NAME already exists. Attaching..."
    tmux attach-session -t "$EXP_NAME"
    exit 0
fi

# Create new session
tmux new-session -d -s "$EXP_NAME"

# Spawns a pane for running QD loop
tmux send-keys -t $EXP_NAME "
export VLA_SERVER_URIs="$(IFS=,; echo "${VLA_SERVER_URIs[*]}")"
uv run -m src.env_search envgen=$EXP_NAME
" C-m

num_servers=${#VLA_SERVER_URIs[@]}
# Spawns a pane for each VLA server; user needs to go to each pane to launch 
# the server manually. The main process (QD) will wait for all VLA servers to 
# be launched
for server_id in $(seq 1 $num_servers)
do
  if ((server_id==1)); then
    tmux split-window -h -t "$EXP_NAME"
    tmux select-pane -t "$EXP_NAME:.1"
  else
    tmux split-window -v
  fi
done

# Make the server panes have equal height
RIGHT_HEIGHT=$(tmux display -p -t "$EXP_NAME:.0" '#{pane_height}')
RIGHT_PANE_HEIGHT=$((RIGHT_HEIGHT / num_servers))
for server_id in $(seq 1 $num_servers)
do
  tmux resize-pane -t "$EXP_NAME:0.$server_id" -y "$RIGHT_PANE_HEIGHT"
done

# Attach
tmux attach-session -t "$EXP_NAME"