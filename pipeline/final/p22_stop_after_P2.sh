#!/bin/bash
# Stop P22 (and its child processes) once it finishes P2 scoring and starts its V1 half; P23 handles V1 faster in parallel.
log="${BDC_ROOT}/experiments/claude_carla_overnight/final/p22_P2V1.log"
pid=$(pgrep -f "^bash p22_P2V1.sh" | head -n 1)
echo "p22 pid $pid"
[ -n "$pid" ] || exit 0
until grep -q "pvrcnn_veh_V1 finished" "$log"; do sleep 10; done
kill_tree() { for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill "$1" 2>/dev/null; }
kill_tree "$pid"
echo "P22 stopped after P2 $(date -Is)" >> "$log"
