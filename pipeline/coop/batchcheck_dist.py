"""Distribution of batch-1 vs batch-4 differences: share of boxes with an exact (<1e-3) counterpart, split by score band."""
import os
import json
from pathlib import Path

import numpy as np

t = Path(os.environ['BDC_ROOT'] + '/experiments/claude_carla_overnight/coop/batchcheck')
a, b = json.loads((t / 'b1.json').read_text()), json.loads((t / 'b4.json').read_text())
bands = {'score>=0.3': [0, 0], '0.1-0.3': [0, 0], '<0.1': [0, 0]}
cls_exact = {}
for g in a:
    for f in a[g]:
        Y = np.array([p['box'] + [p['score']] for p in b[g][f]]).reshape(-1, 8)
        cy = np.array([p['class'] for p in b[g][f]])
        for p in a[g][f]:
            m = cy == p['class']
            ok = bool(m.any() and (np.abs(Y[m] - np.array(p['box'] + [p['score']])).max(1) < 1e-3).any())
            k = 'score>=0.3' if p['score'] >= 0.3 else '0.1-0.3' if p['score'] >= 0.1 else '<0.1'
            bands[k][0] += ok
            bands[k][1] += 1
            c = cls_exact.setdefault(p['class'], [0, 0])
            c[0] += ok
            c[1] += 1
print('exact-match share by score band:', {k: f'{v[0]}/{v[1]}' for k, v in bands.items()})
print('by class:', {k: f'{v[0]}/{v[1]}' for k, v in cls_exact.items()})
