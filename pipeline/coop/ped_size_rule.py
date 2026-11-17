"""Experiment: set each pedestrian box's length/width from the visible point extent instead of the detector's constant 0.75 m.
Takes a finished prediction file (after calibration), rewrites pedestrian L/W, and scores pedestrian AP (all / half A / half B).
Rule: with n >= min_pts non-ground points inside the box grown by g, L' = (1-w)*L + w*clip(extent_x + c, lo, hi); same for W.
usage: ped_size_rule.py <set dir> <pred name>"""
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402
import fuse_test as FT  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
gt = {f: [{'class': g['class'], 'box': g['box']} for g in v if g['points'] >= 5 and g['class'] == 'pedestrian']
      for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
ids = sorted(gt)
halves = [ids, ids[0::2], ids[1::2]]
pred = {f: [x for x in v if x['class'] == 'pedestrian'] for f, v in json.loads((d / f'pred_{name}.json').read_text()).items()}
clouds = FT.load(d / 'scans')
GROW = [0.15, 0.3]
feat = {}
for f in ids:
    p = calib.nonground(np.asarray(clouds[f])[:, :3])
    rows = []
    for x in pred[f]:
        b = np.array(x['box'], float)
        q = p[np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) < 1.5] - b[:3]
        cs, sn = np.cos(-b[6]), np.sin(-b[6])
        lx, ly = q[:, 0] * cs - q[:, 1] * sn, q[:, 0] * sn + q[:, 1] * cs
        r = []
        for g in GROW:
            m = (np.abs(lx) <= b[3] / 2 + g) & (np.abs(ly) <= b[4] / 2 + g) & (np.abs(q[:, 2]) <= b[5] / 2 + 0.15)
            r += [np.ptp(lx[m]) if m.sum() else 0, np.ptp(ly[m]) if m.sum() else 0, m.sum()]
        rows.append(r)
    feat[f] = np.array(rows, float).reshape(-1, 3 * len(GROW))
del clouds


def score(p):
    out = []
    for h in halves:
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: gt[f] for f in h}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 1))
    return out


print('base', score(pred), flush=True)
for gi, c, w, lo, mn in itertools.product(range(len(GROW)), [0.1, 0.2, 0.3], [0.5, 1.0], [0.5, 0.6], [5, 15]):
    new = {}
    for f in ids:
        out = []
        for x, r in zip(pred[f], feat[f]):
            ex, ey, n = r[3 * gi:3 * gi + 3]
            b = list(x['box'])
            if n >= mn:
                b[3] = (1 - w) * b[3] + w * float(np.clip(ex + c, lo, 1.05))
                b[4] = (1 - w) * b[4] + w * float(np.clip(ey + c, lo, 1.05))
            out.append({**x, 'box': b})
        new[f] = out
    print(f'grow {GROW[gi]} c {c} w {w} lo {lo} min_pts {mn}', score(new), flush=True)
print('SIZE_RULE_DONE')
