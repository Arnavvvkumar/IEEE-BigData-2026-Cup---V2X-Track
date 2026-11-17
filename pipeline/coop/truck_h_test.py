"""Test: truck boxes are ~0.18 m too low at the top (zh_diag.py). Variants that raise the truck top; scored on truck AP
(all labels) and genuine-truck AP (TrashCan excluded), all / half A / half B.
usage: truck_h_test.py <set dir> <pred name>"""
import importlib.util
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
gt = {f: [g for g in v if g['points'] >= 5 and g['class'] == 'truck'] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
ids = sorted(gt)
halves = [ids, ids[0::2], ids[1::2]]
pred = {f: [x for x in v if x['class'] == 'truck'] for f, v in json.loads((d / f'pred_{name}.json').read_text()).items()}
clouds = FT.load(d / 'scans')
top = {}
for f in ids:
    p = calib.nonground(np.asarray(clouds[f])[:, :3])
    t = []
    for x in pred[f]:
        q = np.array(x['box'], float)
        loc = p[np.hypot(p[:, 0] - q[0], p[:, 1] - q[1]) < max(q[3], q[4]) / 2 + 0.5] - q[:3]
        cs, sn = np.cos(-q[6]), np.sin(-q[6])
        lx, ly = loc[:, 0] * cs - loc[:, 1] * sn, loc[:, 0] * sn + loc[:, 1] * cs
        m = (np.abs(lx) <= q[3] / 2 + .15) & (np.abs(ly) <= q[4] / 2 + .15) & (np.abs(loc[:, 2]) <= q[5] / 2 + .4)
        t.append((loc[m, 2].max() + q[2], m.sum()) if m.sum() else (0.0, 0))
    top[f] = t
del clouds
cans = {f: [np.array(g['box']) for g in gt[f] if g['type'] == 'TrashCan'] for f in ids}


def ap(p, g):
    out = []
    for h in halves:
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: [{'class': 'truck', 'box': x['box']} for x in g[f]] for f in h}, ['truck'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['truck']['AP_0.5_R40'], 1))
    return out


def score(p):
    pg = {f: [x for x in p[f] if not any(ev._iou_3d(np.array(x['box']), c) >= 0.3 for c in cans[f])] for f in ids}
    return 'truck', ap(p, gt), 'genuine', ap(pg, {f: [g for g in gt[f] if g['type'] != 'TrashCan'] for f in ids})


def shift(dh, dz):
    return {f: [{**x, 'box': x['box'][:2] + [x['box'][2] + dz, x['box'][3], x['box'][4], x['box'][5] + dh, x['box'][6]]} for x in pred[f]] for f in ids}


def from_points(w, c, mn):
    out = {}
    for f in ids:
        new = []
        for x, (tp, n) in zip(pred[f], top[f]):
            b = list(x['box'])
            if n >= mn:
                bot, old = b[2] - b[5] / 2, b[2] + b[5] / 2
                t = (1 - w) * old + w * max(old, tp + c)
                b[2], b[5] = (bot + t) / 2, t - bot
            new.append({**x, 'box': b})
        out[f] = new
    return out


print('base', *score(pred), flush=True)
for dh, dz in [(0.10, 0.05), (0.14, 0.07), (0.18, 0.09), (0.14, 0.11), (0.22, 0.11)]:
    print(f'shift dh {dh} dz {dz}', *score(shift(dh, dz)), flush=True)
for w, c, mn in [(1.0, 0.0, 10), (1.0, 0.05, 10), (0.5, 0.05, 10), (1.0, 0.05, 30)]:
    print(f'points w {w} c {c} min_pts {mn}', *score(from_points(w, c, mn)), flush=True)
print('TRUCK_H_DONE')
