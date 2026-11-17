"""Which filter on the cooperative dev labels reproduces the organizers' test counts (9.40 / 19.06 / 2.48 per frame)?"""
import json
import sys
from pathlib import Path

import numpy as np

gt = json.loads(Path(sys.argv[1]).read_text())
F = len(gt)
C = ['vehicle', 'pedestrian', 'truck']
print('test per frame           vehicle pedestrian truck\n  organizers               9.40 19.06 2.48')
for thr in (0, 1, 2, 3, 5, 10):
    row = [sum(1 for v in gt.values() for g in v if g['class'] == c and g['points'] >= thr) / F for c in C]
    print(f'  coop, points >= {thr:2d}      ', ' '.join(f'{x:5.2f}' for x in row))
for thr in (0, 1, 5):
    row = [sum(1 for v in gt.values() for g in v if g['class'] == c and (g['agent'] == '-1' or g['points'] >= thr)) / F for c in C]
    print(f'  ego + added if pts >= {thr}', ' '.join(f'{x:5.2f}' for x in row))
tc = {}
for v in gt.values():
    for g in v:
        tc.setdefault((g['class'], g['type']), [0, 0])
        tc[(g['class'], g['type'])][0] += 1
        tc[(g['class'], g['type'])][1] += g['points'] >= 5
print('type counts (all, with >=5 points):', {f'{k[0]}/{k[1]}': v for k, v in sorted(tc.items())})
