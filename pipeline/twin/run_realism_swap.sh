#!/bin/bash
# Waits for realism_diag.py to finish, then runs the greedy swap search for 150 minutes (laptop CPU only).
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/twin"
export PYTHONPATH="$HOME/projects/roadside-pilot/realism_deps"
while pgrep -f realism_diag.py > /dev/null; do sleep 20; done
"$HOME/projects/bdc2026-track-b/.env/bin/python" realism_swap.py 150
