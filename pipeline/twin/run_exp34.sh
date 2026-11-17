#!/bin/bash
# Kill the two stuck waiters (their pgrep matched their own command line), then run realism experiments D and C in sequence.
pkill -f "while pgrep -f realism_exp" 2>/dev/null
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/twin"
export PYTHONPATH="$HOME/projects/roadside-pilot/realism_deps"
P="$HOME/projects/bdc2026-track-b/.env/bin/python"
"$P" realism_exp4.py > realism_exp4.log 2>&1
"$P" realism_exp3.py > realism_exp3.log 2>&1
echo EXP34_DONE
