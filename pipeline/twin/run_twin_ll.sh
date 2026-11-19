#!/bin/bash
# Site-twin frames with long, low truck-class objects, then the merged set canon + twin + twin_st + twin_ll, then Lane S:
# SECOND trained on it (vehicle / truck voter). Synthetic-only, from scratch. Laptop only.
set -euo pipefail
here="${BDC_ROOT}/experiments/claude_carla_overnight/twin"
canon="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
P="${BDC_PYTHON}"
export PYTHONPATH="${BDC_REALISM_DEPS}"
cd "$here"
echo "LL_START $(date -Is)"
if [ ! -f "$D/claude_twin_ll_v1/custom_infos_train.pkl" ]; then
  LONGLOW=1 PREFIX=twinll SEED0=700000 "$P" build_twin_st.py --src "$D/claude_canon_all_v1" --pred "$here/static_filter_cw_b_predictions.json" --out "$D/claude_twin_ll_v1" --n 4000 --workers 6 --keep-sensor 0 | tail -n 12
fi
[ -d "$D/claude_canon_all_twin_ll_v1" ] || "$P" merge_sets.py "$D/claude_canon_all_twin_ll_v1" "$D/claude_canon_all_v1" "$D/claude_twin_v1" "$D/claude_twin_st_v1" "$D/claude_twin_ll_v1" | tail -n 5
"$P" - << 'PY'
import numpy as np
from pathlib import Path
D = Path.home() / 'projects/bdc2026-track-b/data/claude_twin_ll_v1/labels'
fs = sorted(D.glob('*.txt'))
s = np.array([[float(x) for x in ln.split()[3:6]] for p in fs[:400] for ln in p.read_text().splitlines() if ln.endswith('truck')])
print('frames', len(fs), 'truck boxes in 400 frames', len(s), 'L/W/H p10/50/90', [np.round(np.percentile(s[:, i], [10, 50, 90]), 2).tolist() for i in range(3)],
      'long-low share', round(float(((s[:, 0] >= 7.3) & (s[:, 2] <= 2.9)).mean()), 2))
PY
echo "LL_DATA_DONE $(date -Is)"
cd "$canon"
export WORKERS=4
bash "$canon/run_arm_par_ns.sh" second_twin_ll_s1 "$canon/second_canon_all_v1.yaml" "$D/claude_canon_all_twin_ll_v1" 12 '-3.15 -3.0 -2.85 -2.7' --use_amp
echo LANES_DONE
