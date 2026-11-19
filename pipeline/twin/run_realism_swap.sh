#!/bin/bash
# Waits for realism_diag.py to finish, then runs the greedy swap search for 150 minutes (laptop CPU only).
cd "${BDC_ROOT}/experiments/claude_carla_overnight/twin"
export PYTHONPATH="${BDC_REALISM_DEPS}"
while pgrep -f realism_diag.py > /dev/null; do sleep 20; done
"${BDC_PYTHON}" realism_swap.py 150
