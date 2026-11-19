#!/bin/bash
# P2: site-twin frames whose synthetic trucks are resized to pickup/van/box-truck size, then merged training set
# (canon + twin + twin-small-trucks). Synthetic-only; real dev clouds only for the unlabelled background, as in twin_v1.
set -euo pipefail
here="${BDC_ROOT}/experiments/claude_carla_overnight/twin"
D="${BDC_DATA}"
P="${BDC_PYTHON}"
export PYTHONPATH="${BDC_REALISM_DEPS}"
cd "$here"
echo "ST_START $(date -Is)"
"$P" build_twin_st.py --src "$D/claude_canon_all_v1" --pred "$here/static_filter_cw_b_predictions.json" --out "$D/claude_twin_st_v1" --n 3000 --workers 5 --keep-sensor 0 | tail -n 14
"$P" merge_sets.py "$D/claude_canon_all_twin_st_v1" "$D/claude_canon_all_v1" "$D/claude_twin_v1" "$D/claude_twin_st_v1" | tail -n 6
echo "ST_DONE $(date -Is)"
