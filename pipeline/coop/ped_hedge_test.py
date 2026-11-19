"""Pedestrian box HEDGING: each pedestrian box is also emitted as a few size / position variants at a lower score (score * f), so that
when the primary box is loose (IoU 0.1-0.5 with the label, 28-43% of pedestrians) one of the variants may reach IoU >= 0.5.
Prediction counts on the leaderboard suggest #1 emits about 720 pedestrian boxes per frame (36,010 on 50 frames).
Scores pedestrian AP (all / half A / half B) and the number of pedestrian boxes per frame.
usage: ped_hedge_test.py <set dir> <norule pred name> [workers]"""
import os
import importlib.util
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
halves = [ids, ids[0::2], ids[1::2]]
gt = {f: [{'class': 'pedestrian', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
pred = {f: [x for x in v if x['class'] == 'pedestrian'] for f, v in json.loads((d / f'pred_{name}.json').read_text()).items()}


def variants(x, kind):
    b = x['box']
    u = np.array(b[:2]) - np.array([-3.9, 0.0])    # direction from the sensor is not needed for the symmetric variants below
    out = []
    if 'size' in kind:
        for sl, sw in ((0.8, 0.8), (1.25, 1.25), (1.0, 1.4), (1.4, 1.0)):
            out.append(b[:3] + [b[3] * sl, b[4] * sw] + b[5:])
    if 'shift' in kind:
        for dx, dy in ((0.15, 0), (-0.15, 0), (0, 0.15), (0, -0.15)):
            out.append([b[0] + dx, b[1] + dy] + b[2:])
    if 'rider' in kind:
        out.append(b[:3] + [1.5, 0.7] + b[5:])
        out.append(b[:3] + [0.6, 1.5] + b[5:])
    if 'tall' in kind:
        out.append(b[:2] + [b[2] + 0.1, b[3], b[4], b[5] * 1.12] + b[6:])
    return out


def run(job):
    label, kind, f, smin = job
    new, n = {}, 0
    for fr in ids:
        o = list(pred[fr])
        for x in pred[fr]:
            if x['score'] >= smin and kind:
                for vb in variants(x, kind):
                    o.append({**x, 'box': vb, 'score': x['score'] * f})
        new[fr] = o
        n += len(o)
    res = []
    for h in halves:
        m = ev.evaluate_3d_detection({fr: new[fr] for fr in h}, {fr: gt[fr] for fr in h}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        res.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 1))
    return f'{label:42s} boxes/frame {n / len(ids):6.0f}  AP {res}'


if __name__ == '__main__':
    jobs = [('base', '', 0, 0), ('size x4, f 0.3', 'size', 0.3, 0), ('size x4, f 0.1', 'size', 0.1, 0), ('size x4, f 0.05', 'size', 0.05, 0),
            ('shift x4, f 0.3', 'shift', 0.3, 0), ('shift x4, f 0.1', 'shift', 0.1, 0),
            ('size+shift x8, f 0.1', 'size shift', 0.1, 0), ('size+shift x8, f 0.05', 'size shift', 0.05, 0),
            ('size+shift, f 0.1, score>=0.1 only', 'size shift', 0.1, 0.1), ('size+shift, f 0.1, score>=0.2 only', 'size shift', 0.1, 0.2),
            ('size+rider, f 0.1', 'size rider', 0.1, 0), ('tall x1, f 0.1', 'tall', 0.1, 0)]
    with get_context('fork').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 5) as pool:
        for line in pool.imap(run, jobs):
            print(line, flush=True)
    print('PED_HEDGE_DONE')
