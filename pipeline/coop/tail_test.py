"""'Guaranteed' tail: boxes appended with scores strictly below every existing box of their class cannot change any existing
match or PR point (official scorer: per-frame greedy by score, global sort, interpolated R40), so AP can only stay or rise.
Measure how much it can add on dev. Base = current build recipe (truck rule a.5 base.1 + dedupe + pedestrian hedging).
usage: tail_test.py <set dir> <norule pred name>"""
import os
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = {f: [{'class': g['class'], 'box': g['box']} for g in v if g['points'] >= 5] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
ids = sorted(G)
C = ['vehicle', 'pedestrian', 'truck']
base = calib.hedge_pedestrians(calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1})))


def curve(p, cls):
    m = ev.evaluate_3d_detection({f: p[f] for f in ids}, {f: G[f] for f in ids}, [cls], iou_thresholds=(0.5,), iou_type='3d')
    return 100 * m['per_class'][cls]['AP_0.5_R40'], m['per_class'][cls]['n_pred']


def recall_stats(p, cls):
    """max recall and precision at the end of the list (re-implements the scorer's matching)"""
    tp = n = ngt = 0
    for f in ids:
        gts = [np.asarray(g['box']) for g in G[f] if g['class'] == cls]; ngt += len(gts)
        used = [False] * len(gts)
        for x in sorted([x for x in p[f] if x['class'] == cls], key=lambda x: -x['score']):
            n += 1; bi, bj = 0, -1
            for j, g in enumerate(gts):
                if not used[j]:
                    i = ev._iou_3d(np.asarray(x['box']), g)
                    if i > bi: bi, bj = i, j
            if bi >= 0.5: used[bj] = True; tp += 1
    return tp / max(ngt, 1), tp / max(n, 1), ngt


def tail(p, cls, variants, smin=0.0):
    """append variants of every box of cls (score >= smin) at scores below the class minimum, ordered by parent score"""
    lo = min(x['score'] for f in ids for x in p[f] if x['class'] == cls)
    out = {f: list(v) for f, v in p.items()}
    for f in ids:
        for x in p[f]:
            if x['class'] != cls or x['score'] < smin:
                continue
            b = x['box']; c, s = np.cos(b[6]), np.sin(b[6])
            for k, (sl, sw, sh, da, dc, dz) in enumerate(variants):
                nb = [b[0] + da * c - dc * s, b[1] + da * s + dc * c, b[2] + dz, b[3] * sl, b[4] * sw, b[5] * sh, b[6]]
                out[f].append({**x, 'box': nb, 'score': lo * 0.5 * x['score'] / (1 + k)})
    return out


for cls in C:
    ap0, n0 = curve(base, cls)
    r, pr, ngt = recall_stats(base, cls)
    print(f'{d.name} {cls:10s} AP {ap0:.2f} | n_pred {n0} n_gt {ngt} | max recall {r:.3f} | precision at end {pr:.3f}', flush=True)

SETS = {
    'pedestrian': [(1.3, 1.3, 1, 0, 0, 0), (0.75, 0.75, 1, 0, 0, 0), (1, 1, 1, 0.25, 0, 0), (1, 1, 1, -0.25, 0, 0), (1, 1, 1, 0, 0.25, 0),
                   (1, 1, 1, 0, -0.25, 0), (1, 1, 1.2, 0, 0, 0.1), (1, 1, 0.85, 0, 0, -0.1)],
    'vehicle': [(1.1, 1.1, 1, 0, 0, 0), (0.9, 0.9, 1, 0, 0, 0), (1, 1, 1, 0.5, 0, 0), (1, 1, 1, -0.5, 0, 0), (1, 1, 1, 0, 0.3, 0), (1, 1, 1, 0, -0.3, 0)],
    'truck': [(1.15, 1.05, 1, 0, 0, 0), (0.85, 1, 1, 0, 0, 0), (1, 1, 1, 0.8, 0, 0), (1, 1, 1, -0.8, 0, 0), (1, 1, 1.15, 0, 0, 0.2)],
}
for cls, v in SETS.items():
    t = tail(base, cls, v, smin=0.01)
    ap1, n1 = curve(t, cls)
    ap0, n0 = curve(base, cls)
    print(f'{d.name} TAIL {cls:10s} AP {ap0:.2f} -> {ap1:.2f} (+{ap1 - ap0:.2f}) | boxes {n0} -> {n1}', flush=True)
