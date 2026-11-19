"""REPORT-ONLY (the user said leave trucks alone): does the scan-point rule help trucks? Truck boxes (after the truck rule + dedupe, i.e. the
final truck set) with < thr non-ground points inside (box grown 0.1 m) get score x m. thr in {10, 30, 100}, m in {0.5, 0.25}.
Metrics as in truck_real.py: REAL (non-parked), Truck+Van, all-truck. Adoption would need all three up on BOTH dev sets.
usage: truck_minpts.py <set dir> <full norule pred name>"""
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
d, name = Path(sys.argv[1]), sys.argv[2]
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
        out.append(round(100 * m['per_class']['truck']['AP_0.5_R40'], 2))
    return out


clouds = {f: np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
pred = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))
T = {f: [x for x in pred[f] if x['class'] == 'truck'] for f in ids}
N = {}
for f in ids:
    q = calib.nonground(clouds[f])
    N[f] = []
    for x in T[f]:
        b = np.array(x['box'], float)
        near = np.hypot(q[:, 0] - b[0], q[:, 1] - b[1]) < np.hypot(b[3], b[4]) / 2 + 0.5
        r = q[near] - b[:3]
        cs, sn = np.cos(-b[6]), np.sin(-b[6])
        xx = r[:, 0] * cs - r[:, 1] * sn; yy = r[:, 0] * sn + r[:, 1] * cs
        N[f].append(int(((np.abs(xx) <= b[3] / 2 + 0.1) & (np.abs(yy) <= b[4] / 2 + 0.1) & (np.abs(r[:, 2]) <= b[5] / 2 + 0.1)).sum()))
allN = np.array([n for f in ids for n in N[f]])
print(f'{d.name} truck boxes {len(allN)} | points inside q10/q25/q50/q75 {np.percentile(allN, [10, 25, 50, 75]).round(0).tolist()} | <10: {(allN < 10).mean():.2f} <30: {(allN < 30).mean():.2f} <100: {(allN < 100).mean():.2f}', flush=True)
for label, thr, m in [('base', 0, 1.0)] + [(f'pts<{t} x{mm}', t, mm) for t in (10, 30, 100) for mm in (0.5, 0.25)]:
    P = {f: [({**x, 'score': x['score'] * m} if n < thr else x) for x, n in zip(T[f], N[f])] for f in ids}
    print(f'{d.name} TRUCK {label:14s} REAL {ap_cond(P, lambda g: not g["static"])} | Truck+Van {ap_cond(P, lambda g: g["type"] in ("Truck", "Van"))} '
          f'| all-truck {ap_cond(P, lambda g: True)}', flush=True)
print('TRUCK_MINPTS_DONE')
