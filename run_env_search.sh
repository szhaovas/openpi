#!/usr/bin/env bash

NUM_SERVERS=3 # This should be the same as cfg.eval.task_eval.num_trials_per_sol
SERVER_PORT_START=8001
GPU_ID_START=3
SESSION=openpi-libero

# Check if session exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists. Attaching..."
    tmux attach-session -t "$SESSION"
    exit 0
fi

# Create new session
tmux new-session -d -s "$SESSION"

# Spawns a pane for running QD loop
tmux send-keys -t $SESSION "
export NUM_SERVERS=$NUM_SERVERS
export SERVER_PORT_START=$SERVER_PORT_START
uv run -m src.env_search
" C-m

# Spawns NUM_SERVERS panes each calling serve_policy.py with its own GPU
for server_id in $(seq 0 $((NUM_SERVERS-1)))
do
  if ((server_id==0)); then
    tmux split-window -h -t "$SESSION"
    tmux select-pane -t "$SESSION:.1"
  else
    tmux split-window -v
  fi
  tmux send-keys -t "$SESSION:0.$((server_id+1))" "
    cd openpi
    CUDA_VISIBLE_DEVICES=$((GPU_ID_START+server_id)) uv run scripts/serve_policy.py --env LIBERO --port $((SERVER_PORT_START+server_id))
  " C-m
done

# Make the server panes have equal height
RIGHT_HEIGHT=$(tmux display -p -t "$SESSION:.0" '#{pane_height}')
RIGHT_PANE_HEIGHT=$((RIGHT_HEIGHT / NUM_SERVERS))
for server_id in $(seq 1 "$NUM_SERVERS")
do
  tmux resize-pane -t "$SESSION:0.$server_id" -y "$RIGHT_PANE_HEIGHT"
done

# Attach
tmux attach-session -t "$SESSION"