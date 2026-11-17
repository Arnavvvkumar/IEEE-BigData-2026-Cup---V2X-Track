#!/bin/bash
# Smoke test: 60 twin frames, then realism of 50 of them (sensor frame) against the real dev frames.
set -euo pipefail
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/twin"
D="$HOME/projects/bdc2026-track-b/data"
P="$HOME/projects/bdc2026-track-b/.env/bin/python"
export PYTHONPATH="$HOME/projects/roadside-pilot/realism_deps"
cd "$here"
rm -rf "$D/claude_twin_smoke"
"$P" build_twin.py --src "$D/claude_canon_all_v1" --pred "$here/static_filter_cw_b_predictions.json" --out "$D/claude_twin_smoke" --n 60 --workers 4 --keep-sensor 60
"$P" twin_realism.py "$D/claude_twin_smoke" 50
