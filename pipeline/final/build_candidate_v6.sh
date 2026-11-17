#!/bin/bash
# Candidate v6 = v5 vehicles and trucks + the v3 pedestrian voters (leaderboard: v5 pedestrian 35.20 vs v3 35.99; vehicle 74.30 vs 69.07).
# Made from build_candidate_v5.sh by dropping vnx010_twin_s2 (pedestrian-only) and using vnx010_pv_ft2 for vehicles only.
# Then checks that the pedestrian boxes equal candidate_v3's (same boxes, scores differ only by the constant pass-count factor).
set -euo pipefail
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final"
P="$HOME/projects/detector-program/env310/bin/python"
sed -e 's/candidate_v5/candidate_v6/g' -e 's/submission_v5/submission_v6/g' -e 's/CANDIDATE_V5_DONE/CANDIDATE_V6_DONE/' \
    -e 's/ vnx010_twin_s2_ep12:pedestrian / /' -e 's/vnx010_pv_ft2_ep5:pedestrian,vehicle/vnx010_pv_ft2_ep5:vehicle/' \
    "$here/build_candidate_v5.sh" > "$here/_build_v6_gen.sh"
grep -c 'vnx010_twin_s2_ep12' "$here/_build_v6_gen.sh" || true
bash "$here/_build_v6_gen.sh" 2>&1 | grep -E 'passes|rule|format|zip |scorer|entries|DONE|missing'
"$P" - "$here" << 'PY'
import json, sys
from pathlib import Path
import numpy as np
h = Path(sys.argv[1])
P = {v: json.loads((h / v / 'predictions.json').read_text()) for v in ('candidate_v3', 'candidate_v5', 'candidate_v6')}
def cls(v, c):
    return {f: sorted([(x['score'], *x['box']) for x in P[v][f] if x['class'] == c], reverse=True) for f in P[v]}
a, b = cls('candidate_v3', 'pedestrian'), cls('candidate_v6', 'pedestrian')
same_n = all(len(a[f]) == len(b[f]) for f in a)
print('pedestrian v6 vs v3: same counts', same_n)
if same_n:
    A = np.array([r for f in a for r in a[f]]); B = np.array([r for f in b for r in b[f]])
    ratio = B[:, 0] / np.maximum(A[:, 0], 1e-12)
    print('  max box difference', float(np.abs(A[:, 1:] - B[:, 1:]).max()), '| score ratio min/max', float(ratio.min()), float(ratio.max()))
    order = all([r[1:] for r in a[f]] == [r[1:] for r in b[f]] for f in a)
    print('  same ranking in every frame', order)
for c in ('vehicle', 'truck'):
    x, y = cls('candidate_v5', c), cls('candidate_v6', c)
    n = all(len(x[f]) == len(y[f]) for f in x)
    d = max(abs(p[i] - q[i]) for f in x for p, q in zip(x[f], y[f]) for i in range(1, 8)) if n else None
    print(f'{c} v6 vs v5: same counts {n} | max box difference {d}')
PY
