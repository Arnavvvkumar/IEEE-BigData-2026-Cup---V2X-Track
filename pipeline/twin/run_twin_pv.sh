#!/bin/bash
# Site-twin frames with size-varied pedestrians (and small trucks), then the merged set canon + twin + twin_st + twin_pv.
set -euo pipefail
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/twin"
D="$HOME/projects/bdc2026-track-b/data"
P="$HOME/projects/bdc2026-track-b/.env/bin/python"
export PYTHONPATH="$HOME/projects/roadside-pilot/realism_deps"
cd "$here"
echo "PV_START $(date -Is)"
PEDVARY=1 PREFIX=twinpv SEED0=500000 "$P" build_twin_st.py --src "$D/claude_canon_all_v1" --pred "$here/static_filter_cw_b_predictions.json" --out "$D/claude_twin_pv_v1" --n 5000 --workers 5 --keep-sensor 0 | tail -n 12
"$P" merge_sets.py "$D/claude_canon_all_twin_pv_v1" "$D/claude_canon_all_v1" "$D/claude_twin_v1" "$D/claude_twin_st_v1" "$D/claude_twin_pv_v1" | tail -n 5
"$P" - << 'PY'
import numpy as np
from pathlib import Path
D = Path.home() / 'projects/bdc2026-track-b/data/claude_twin_pv_v1/labels'
s = np.array([[float(x) for x in ln.split()[3:6]] for p in sorted(D.glob('*.txt'))[:200] for ln in p.read_text().splitlines() if ln.endswith('pedestrian')])
print('pedestrian boxes', len(s), 'L/W/H p10/50/90', [np.round(np.percentile(s[:, i], [10, 50, 90]), 2).tolist() for i in range(3)])
PY
echo "PV_DONE $(date -Is)"
