"""Realistic truck metric (dev trucks that are NOT the same parked object repeated within the scenario; parked repeats = do-not-care).
Also reports all-truck (old gate), Truck+Van type AP, and vehicle AP (to check nothing else moves).
usage: truck_real.py <set dir> <norule pred name> [...]"""
import os
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
d = Path(sys.argv[1])
G = {f: [g for g in v if g['points'] >= 5] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
idx = json.loads((d / 'index.json').read_text())
ids = sorted(G)
by_scen = defaultdict(list)
for f in ids:
    by_scen[idx[f]['scenario']].append(f)
for f in ids:
    for g in G[f]:
        if g['class'] == 'truck':
            others = [h for o in by_scen[idx[f]['scenario']] if o != f for h in G[o] if h['class'] == 'truck' and h['type'] == g['type']]
            g['static'] = any(np.hypot(h['box'][0] - g['box'][0], h['box'][1] - g['box'][1]) < 1.0 for h in others)


def inside(x, g, grow=1.0):
    b = g['box']; q = np.array(x['box'][:2]) - b[:2]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    return abs(q[0] * cs - q[1] * sn) <= b[3] / 2 + grow and abs(q[0] * sn + q[1] * cs) <= b[4] / 2 + grow


def ap_cond(P, keep):
    gt = {f: [{'class': 'truck', 'box': g['box']} for g in G[f] if g['class'] == 'truck' and keep(g)] for f in ids}
    drop = {f: [g for g in G[f] if g['class'] == 'truck' and not keep(g)] for f in ids}
    pr = {f: [x for x in P[f] if not any(inside(x, g) for g in drop[f])] for f in ids}
    out = []
    for h in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: pr[f] for f in h}, {f: gt[f] for f in h}, ['truck'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['truck']['AP_0.5_R40'], 1))
    return out


for name in sys.argv[2:]:
    p = d / f'pred_{name}.json'
    if not p.exists():
        print(f'{d.name} {name}: MISSING'); continue
    pred = calib.truck_dedupe(calib.truck_rule(json.loads(p.read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))
    P = {f: [x for x in pred[f] if x['class'] == 'truck'] for f in ids}
    real = ap_cond(P, lambda g: not g['static'])
    tv = ap_cond(P, lambda g: g['type'] in ('Truck', 'Van'))
    alltr = ap_cond(P, lambda g: True)
    print(f'{d.name} {name:38s} REAL(non-parked) {real} | Truck+Van {tv} | all-truck {alltr}', flush=True)
