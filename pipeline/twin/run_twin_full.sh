#!/bin/bash
# Full twin set (5,000 frames), realism of its first 50 scans, then the merged training set (canonical full set + twin).
set -euo pipefail
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/twin"
D="$HOME/projects/bdc2026-track-b/data"
P="$HOME/projects/bdc2026-track-b/.env/bin/python"
export PYTHONPATH="$HOME/projects/roadside-pilot/realism_deps"
cd "$here"
"$P" build_twin.py --src "$D/claude_canon_all_v1" --pred "$here/static_filter_cw_b_predictions.json" --out "$D/claude_twin_v1" --n 5000 --workers 5 --keep-sensor 400 | tail -n 14
"$P" merge_sets.py "$D/claude_canon_all_twin_v1" "$D/claude_canon_all_v1" "$D/claude_twin_v1"
"$P" twin_realism.py "$D/claude_twin_v1" 50 | head -n 8
echo TWIN_FULL_DONE
