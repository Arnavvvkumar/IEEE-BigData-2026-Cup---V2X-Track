"""Post-fusion duplicate removal test (CPU): within a class, drop a box whose BEV centre lies inside a higher-scored box of the
same class (or 3D IoU >= thr). Also: drop a truck box sitting on a higher-scored vehicle box only if its score is below r * vehicle score.
usage: dedupe_test.py <set dir> <pred json name>"""
import os
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d = Path(sys.argv[1])
pred = json.loads((d / sys.argv[2]).read_text())
gt = {f: [g for g in v if g['points'] >= 5] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
ids = sorted(gt)
halves = [ids, ids[0::2], ids[1::2]]


def inside(c, b, shrink=1.0):
    q = c - b[:2]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    return abs(q[0] * cs - q[1] * sn) <= b[3] / 2 * shrink and abs(q[0] * sn + q[1] * cs) <= b[4] / 2 * shrink


def dedupe(p, cls, shrink):
    out = {}
    for f in ids:
        keep, kept = [], []
        for x in sorted(p[f], key=lambda x: -x['score']):
            if x['class'] == cls:
                b = np.array(x['box'], float)
                if any(inside(b[:2], kb, shrink) for kb in kept):
                    continue
                kept.append(b)
            keep.append(x)
        out[f] = keep
    return out


def ap(p, cls):
    r = []
    for h in halves:
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: [{'class': x['class'], 'box': x['box']} for x in gt[f]] for f in h}, [cls], iou_thresholds=(0.5,), iou_type='3d')
        r.append(round(100 * m['per_class'][cls]['AP_0.5_R40'], 1))
    return r


cans = {f: [np.array(g['box']) for g in gt[f] if g['type'] == 'TrashCan'] for f in ids}


def genuine(p):
    pg = {f: [x for x in p[f] if not (x['class'] == 'truck' and any(ev._iou_3d(np.array(x['box']), c) >= 0.3 for c in cans[f]))] for f in ids}
    g2 = {f: [g for g in gt[f] if g['type'] != 'TrashCan'] for f in ids}
    r = []
    for h in halves:
        m = ev.evaluate_3d_detection({f: pg[f] for f in h}, {f: [{'class': x['class'], 'box': x['box']} for x in g2[f]] for f in h}, ['truck'], iou_thresholds=(0.5,), iou_type='3d')
        r.append(round(100 * m['per_class']['truck']['AP_0.5_R40'], 1))
    return r


print('base        truck', ap(pred, 'truck'), 'genuine', genuine(pred), 'vehicle', ap(pred, 'vehicle'))
for s in (1.0, 0.8, 0.6):
    q = dedupe(pred, 'truck', s)
    print(f'truck dedupe shrink {s}: truck', ap(q, 'truck'), 'genuine', genuine(q))
for s in (1.0, 0.7):
    q = dedupe(pred, 'vehicle', s)
    print(f'vehicle dedupe shrink {s}: vehicle', ap(q, 'vehicle'))
