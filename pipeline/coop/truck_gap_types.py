"""Truck dev-vs-test gap, step 1: what is a 'truck' in the dev labels? Types, sizes, point counts, range; and V2X-Real class map.
usage: truck_gap_types.py   (reads dev150 and dev350 gt_coop.json; no test labels exist locally)"""
import os
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'V2X-Real'))
from opencood.data_utils import SUPER_CLASS_MAP  # noqa: E402

print('SUPER_CLASS_MAP:', {k: v for k, v in SUPER_CLASS_MAP.items()})
base = W / 'experiments/claude_carla_overnight/coop'
for s in ('dev150', 'dev350'):
    G = json.loads((base / s / 'gt_coop.json').read_text())
    F = len(G)
    tr = [g for v in G.values() for g in v if g['class'] == 'truck']
    print(f'\n{s}: {F} frames | truck labels {len(tr)} ({len(tr) / F:.2f}/frame) | with >=5 pts {sum(g["points"] >= 5 for g in tr)} ({sum(g["points"] >= 5 for g in tr) / F:.2f}/frame)')
    for c in ('vehicle', 'pedestrian'):
        a = [g for v in G.values() for g in v if g['class'] == c]
        print(f'  {c}: all {len(a) / F:.2f}/frame, >=5 pts {sum(g["points"] >= 5 for g in a) / F:.2f}/frame')
    print('  truck types (all / >=5 pts):', {t: (n, sum(1 for g in tr if g['type'] == t and g['points'] >= 5)) for t, n in Counter(g['type'] for g in tr).most_common()})
    for t in sorted(set(g['type'] for g in tr)):
        x = np.array([g['box'] for g in tr if g['type'] == t])
        r = np.hypot(x[:, 0], x[:, 1])
        pts = np.array([g['points'] for g in tr if g['type'] == t])
        print(f'   {t:14s} n={len(x):4d} L med {np.median(x[:, 3]):5.2f} [{x[:, 3].min():.1f}-{x[:, 3].max():.1f}]  W {np.median(x[:, 4]):.2f}  H {np.median(x[:, 5]):.2f}'
              f'  range med {np.median(r):5.1f} | pts med {np.median(pts):.0f}, 0pts {(pts == 0).mean():.0%}, <5 {(pts < 5).mean():.0%}')
    frames_with_truck = sum(1 for v in G.values() if any(g['class'] == 'truck' and g['points'] >= 5 for g in v))
    print(f'  frames with >=1 truck (>=5 pts): {frames_with_truck}/{F}')
