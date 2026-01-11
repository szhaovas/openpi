#!/usr/bin/env bash

NUM_SERVERS=4 # This should be the same as cfg["eval"]["num_trials_per_sol"]
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
cd \"$(pwd)\"
source examples/libero/.venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/libero
export NUM_SERVERS=$NUM_SERVERS
python -m experiments.qd_search
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
    cd \"$(pwd)\"
    CUDA_VISIBLE_DEVICES=$server_id uv run scripts/serve_policy.py --env LIBERO --port $((8000+server_id))
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