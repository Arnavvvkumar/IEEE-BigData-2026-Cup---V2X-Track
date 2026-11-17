"""Diagnostic (no AP run): could box top/bottom be read from the points better than the detector predicts them?
For each label (>=5 pts) matched to the nearest same-class prediction (centre within r, score >= 0.1):
  error of predicted top / bottom vs label, and of two point-based estimates:
  top_pts = highest non-ground point in the predicted footprint (grown 0.15 m), bottom_gnd = fixed ground plane under the box centre.
usage: zh_diag.py <set dir> <pred name>"""
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402
import fuse_test as FT  # noqa: E402

d, name = Path(sys.argv[1]), sys.argv[2]
gt = json.loads((d / 'gt_coop.json').read_text())
pred = json.loads((d / f'pred_{name}.json').read_text())
clouds = FT.load(d / 'scans')
R = {'pedestrian': 0.5, 'vehicle': 1.5, 'truck': 2.0}
rows = {c: [] for c in R}
for f, v in gt.items():
    p = calib.nonground(np.asarray(clouds[f])[:, :3])
    for c, r in R.items():
        P = [x for x in pred[f] if x['class'] == c and x['score'] >= 0.1]
        if not P:
            continue
        pb = np.array([x['box'] for x in P], float)
        for g in v:
            if g['class'] != c or g['points'] < 5 or g.get('type') == 'TrashCan':
                continue
            b = np.array(g['box'], float)
            dist = np.hypot(pb[:, 0] - b[0], pb[:, 1] - b[1])
            i = int(dist.argmin())
            if dist[i] > r:
                continue
            q = pb[i]
            loc = p[np.hypot(p[:, 0] - q[0], p[:, 1] - q[1]) < max(q[3], q[4]) / 2 + 0.5] - q[:3]
            cs, sn = np.cos(-q[6]), np.sin(-q[6])
            x, y = loc[:, 0] * cs - loc[:, 1] * sn, loc[:, 0] * sn + loc[:, 1] * cs
            m = (np.abs(x) <= q[3] / 2 + .15) & (np.abs(y) <= q[4] / 2 + .15) & (np.abs(loc[:, 2]) <= q[5] / 2 + .4)
            if m.sum() < 3:
                continue
            zg = q[0] * calib.PLANE[0] + q[1] * calib.PLANE[1] + calib.PLANE[2]
            rows[c].append((q[2] + q[5] / 2 - (b[2] + b[5] / 2), q[2] - q[5] / 2 - (b[2] - b[5] / 2),
                            loc[m, 2].max() + q[2] - (b[2] + b[5] / 2), zg - (b[2] - b[5] / 2), q[5] - b[5], m.sum(), np.hypot(b[0], b[1])))
for c, v in rows.items():
    a = np.array(v)
    if not len(a):
        continue
    print(f'== {c}: matched {len(a)}')
    for k, nm in enumerate(['pred top - label top', 'pred bottom - label bottom', 'highest point - label top', 'ground plane - label bottom', 'pred height - label height']):
        e = a[:, k]
        print(f'  {nm:28s} median {np.median(e):+.3f}  spread(p10..p90) {np.percentile(e, 10):+.3f}..{np.percentile(e, 90):+.3f}  mean|err after removing median| {np.abs(e - np.median(e)).mean():.3f}')
    for lo, hi in [(0, 25), (25, 45), (45, 200)]:
        m = (a[:, 6] >= lo) & (a[:, 6] < hi)
        if m.sum() > 20:
            print(f'  range {lo}-{hi} m n {int(m.sum())}: pred top {np.median(a[m, 0]):+.3f} bottom {np.median(a[m, 1]):+.3f} | point top {np.median(a[m, 2]):+.3f} ground bottom {np.median(a[m, 3]):+.3f}')
