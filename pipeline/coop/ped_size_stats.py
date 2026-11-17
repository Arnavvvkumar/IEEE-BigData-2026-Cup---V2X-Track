"""Spread of predicted pedestrian box length/width for one model's raw predictions.
usage: ped_size_stats.py <set dir> <tag_epN> [...]"""
import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
for tag in sys.argv[2:]:
    data = json.loads((d / 'allpred' / f'{tag}_allpred.json').read_text())
    b = np.array([x['box'] for p in data.values() for f in p for x in p[f] if x['class'] == 'pedestrian' and x['score'] >= 0.3])
    if not len(b):
        print(tag, 'no pedestrian boxes')
        continue
    q = lambda v: [round(float(t), 2) for t in np.percentile(v, [10, 50, 90])]
    print(tag, 'n', len(b), 'L', q(b[:, 3]), 'W', q(b[:, 4]), 'H', q(b[:, 5]), flush=True)
