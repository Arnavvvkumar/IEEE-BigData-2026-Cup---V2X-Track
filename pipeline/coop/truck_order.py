"""Codex idea (Nov 15 PM): apply the truck point penalty BEFORE truck_dedupe, so a sparse high-scored box cannot delete a better-supported
overlapping box before its own score is corrected. Pre-registered variants, production order otherwise (calibrate -> truck_rule a0.5/base0.1
-> [dedupe/minpts] -> truck_boost_hedge boost 1.0):
  BASE  rule -> dedupe                       (v17 trucks)
  REF   rule -> dedupe -> minpts(30, x0.25)  (v18 as built)
  PRE   rule -> minpts(30, x0.25) -> dedupe  (Codex)
Also an upper-bound audit: in REF, boxes deleted by a higher-scored SPARSE (<30 pts) box while themselves having >= 30 pts.
Metrics as truck_minpts.py: REAL (non-parked), Truck+Van, all-truck, each [full, half A, half B].
usage: truck_order.py <set dir> <full norule pred name>"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
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
    pr = {f: [x for x in P[f] if x['class'] == 'truck' and not any(inside(x, g) for g in drop[f])] for f in ids}
    out = []
    for h in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: pr[f] for f in h}, {f: gt[f] for f in h}, ['truck'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['truck']['AP_0.5_R40'], 3))
    return out


def npts(q, x, grow=0.1):
    b = np.array(x['box'], float)
    near = np.hypot(q[:, 0] - b[0], q[:, 1] - b[1]) < np.hypot(b[3], b[4]) / 2 + 0.5
    r = q[near] - b[:3]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    xx = r[:, 0] * cs - r[:, 1] * sn; yy = r[:, 0] * sn + r[:, 1] * cs
    return int(((np.abs(xx) <= b[3] / 2 + grow) & (np.abs(yy) <= b[4] / 2 + grow) & (np.abs(r[:, 2]) <= b[5] / 2 + grow)).sum())


clouds = {f: np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
raw = calib.calibrate(json.loads((d / f'pred_{name}.json').read_text()), clouds)
ruled = calib.truck_rule(raw, **{**calib.TRUCK, 'a': 0.5, 'base': 0.1})
# keep only trucks from here on (vehicles/pedestrians do not affect truck AP or truck dedupe)
ruled = {f: [x for x in v if x['class'] == 'truck'] for f, v in ruled.items()}

# upper-bound audit on the REF order
nlost, nlost_tp = 0, 0
for f in ids:
    q = calib.nonground(clouds[f])
    kept = []
    for x in sorted(ruled[f], key=lambda x: -x['score']):
        b = np.array(x['box'], float)
        sup = None
        for kx in kept:
            kb = np.array(kx['box'], float); r = b[:2] - kb[:2]
            cs, sn = np.cos(-kb[6]), np.sin(-kb[6])
            if abs(r[0] * cs - r[1] * sn) <= kb[3] / 2 * 0.8 and abs(r[0] * sn + r[1] * cs) <= kb[4] / 2 * 0.8:
                sup = kx; break
        if sup is None:
            kept.append(x); continue
        if npts(q, sup) < 30 <= npts(q, x):
            nlost += 1
            gts = [g for g in G[f] if g['class'] == 'truck']
            if any(inside(x, g, grow=0.0) for g in gts):
                nlost_tp += 1
print(f'{d.name} AUDIT: boxes deleted by a sparser higher-scored box (sup<30, self>=30): {nlost}; of these with centre inside a GT truck: {nlost_tp}', flush=True)

V = {
    'BASE': calib.truck_dedupe(ruled),
    'REF': calib.truck_min_points(calib.truck_dedupe(ruled), clouds),
    'PRE': calib.truck_dedupe(calib.truck_min_points(ruled, clouds)),
}
for k, P in V.items():
    P = calib.truck_boost_hedge(P, clouds, boost=1.0)
    print(f'{d.name} {k:5s} boxes {sum(map(len, P.values()))} | REAL {ap_cond(P, lambda g: not g["static"])} | Truck+Van {ap_cond(P, lambda g: g["type"] in ("Truck", "Van"))} '
          f'| all-truck {ap_cond(P, lambda g: True)}', flush=True)
print('TRUCK_ORDER_DONE')
