"""Diagnostic: do real pedestrian label sizes follow the visible point extent? (decides whether size can be read from points)
For each labelled pedestrian (>=5 pts): points inside the label box grown by 0.15 m; extent along the label's length / width axes and height.
usage: ped_extent.py <set dir>"""
import os
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import fuse_test as FT  # noqa: E402

d = Path(sys.argv[1])
gt = json.loads((d / 'gt_coop.json').read_text())
clouds = FT.load(d / 'scans')
rows = []
for f, v in gt.items():
    p = np.asarray(clouds[f])[:, :3]
    for g in v:
        if g['class'] != 'pedestrian' or g['points'] < 5:
            continue
        b = np.array(g['box'], float)
        q = p[np.abs(p[:, 0] - b[0]) + np.abs(p[:, 1] - b[1]) < 3] - b[:3]
        cs, sn = np.cos(-b[6]), np.sin(-b[6])
        x, y = q[:, 0] * cs - q[:, 1] * sn, q[:, 0] * sn + q[:, 1] * cs
        m = (np.abs(x) <= b[3] / 2 + .15) & (np.abs(y) <= b[4] / 2 + .15) & (np.abs(q[:, 2]) <= b[5] / 2 + .15)
        if m.sum() < 5:
            continue
        rows.append((b[3], b[4], b[5], np.ptp(x[m]), np.ptp(y[m]), np.ptp(q[m, 2]), m.sum(), g.get('type', ''), q[m, 2].max() + b[2], b[2] + b[5] / 2))
R = np.array([r[:7] for r in rows], float)
print('labels used', len(R))
for lo, hi in [(5, 20), (20, 50), (50, 100000)]:
    m = (R[:, 6] >= lo) & (R[:, 6] < hi)
    c = lambda a, b: round(float(np.corrcoef(R[m, a], R[m, b])[0, 1]), 2)
    print(f'points {lo}-{hi}: n {int(m.sum())} | corr(label L, extent L) {c(0, 3)} | corr(label W, extent W) {c(1, 4)} | corr(label H, extent H) {c(2, 5)} | corr(label L, label W) {c(0, 1)}')
    print('   label minus extent, median: L', round(float(np.median(R[m, 0] - R[m, 3])), 3), 'W', round(float(np.median(R[m, 1] - R[m, 4])), 3), 'H', round(float(np.median(R[m, 2] - R[m, 5])), 3))
t = np.array([r[7] for r in rows])
for k in sorted(set(t)):
    m = t == k
    print(f'{k:16s} n {int(m.sum()):5d} label L/W/H median {np.round(np.median(R[m, :3], 0), 2).tolist()}')
top = np.array([r[9] - r[8] for r in rows])
print('label top minus highest point: p10/50/90', np.round(np.percentile(top, [10, 50, 90]), 3).tolist())
