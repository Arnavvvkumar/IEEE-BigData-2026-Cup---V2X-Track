#!/bin/bash
# Smoke test: 60 twin frames, then realism of 50 of them (sensor frame) against the real dev frames.
set -euo pipefail
here="${BDC_ROOT}/experiments/claude_carla_overnight/twin"
D="${BDC_DATA}"
P="${BDC_PYTHON}"
export PYTHONPATH="${BDC_REALISM_DEPS}"
cd "$here"
rm -rf "$D/claude_twin_smoke"
"$P" build_twin.py --src "$D/claude_canon_all_v1" --pred "$here/static_filter_cw_b_predictions.json" --out "$D/claude_twin_smoke" --n 60 --workers 4 --keep-sensor 60
"$P" twin_realism.py "$D/claude_twin_smoke" 50
