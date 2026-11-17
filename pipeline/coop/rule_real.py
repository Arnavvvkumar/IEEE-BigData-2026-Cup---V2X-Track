"""Truck rule retune on the REALISTIC truck metrics (non-parked trucks; Truck+Van types), plus all-truck and vehicle as guards.
Pre-registered grid. Input: norule prediction (current v15s-equivalent list).
usage: rule_real.py <set dir> <norule pred name> [workers]"""
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
halves = (ids, ids[0::2], ids[1::2])
by_scen = defaultdict(list)
for f in ids:
    by_scen[idx[f]['scenario']].append(f)
for f in ids:
    for g in G[f]:
        if g['class'] == 'truck':
            others = [h for o in by_scen[idx[f]['scenario']] if o != f for h in G[o] if h['class'] == 'truck' and h['type'] == g['type']]
            g['static'] = any(np.hypot(h['box'][0] - g['box'][0], h['box'][1] - g['box'][1]) < 1.0 for h in others)
raw = {f: [x for x in v if x['class'] != 'pedestrian'] for f, v in json.loads((d / f'pred_{name}.json').read_text()).items()}


def inside(x, g, grow=1.0):
    b = g['box']; q = np.array(x['box'][:2]) - b[:2]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    return abs(q[0] * cs - q[1] * sn) <= b[3] / 2 + grow and abs(q[0] * sn + q[1] * cs) <= b[4] / 2 + grow


def ap(pr, gt, cls):
    out = []
    for h in halves:
        m = ev.evaluate_3d_detection({f: pr[f] for f in h}, {f: gt[f] for f in h}, [cls], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class'][cls]['AP_0.5_R40'], 1))
    return out


def tr_ap(P, keep):
    gt = {f: [{'class': 'truck', 'box': g['box']} for g in G[f] if g['class'] == 'truck' and keep(g)] for f in ids}
    drop = {f: [g for g in G[f] if g['class'] == 'truck' and not keep(g)] for f in ids}
    pr = {f: [x for x in P[f] if not any(inside(x, g) for g in drop[f])] for f in ids}
    return ap(pr, gt, 'truck')


VGT = {f: [{'class': 'vehicle', 'box': g['box']} for g in G[f] if g['class'] == 'vehicle'] for f in ids}


def run(job):
    label, kw = job
    p = calib.truck_dedupe(calib.truck_rule(raw, **{**calib.TRUCK, 'a': 0.5, 'base': 0.1, **kw}))
    P = {f: [x for x in p[f] if x['class'] == 'truck'] for f in ids}
    V = {f: [x for x in p[f] if x['class'] == 'vehicle'] for f in ids}
    return (f'{d.name} {label:26s} REAL {tr_ap(P, lambda g: not g["static"])} | TV {tr_ap(P, lambda g: g["type"] in ("Truck", "Van"))}'
            f' | all {tr_ap(P, lambda g: True)} | veh {ap(V, VGT, "vehicle")}')


JOBS = [('current a.5 base.1', {})]
for base in (0.07, 0.05, 0.03, 0.02, 0.0):
    JOBS.append((f'base {base}', dict(base=base)))
for a in (0.7, 1.0):
    for base in (0.1, 0.05, 0.03):
        JOBS.append((f'a {a} base {base}', dict(a=a, base=base)))
for kw in (dict(L=4.8), dict(Wd=2.1), dict(H=1.9), dict(L=4.8, Wd=2.1, H=1.9)):
    for base in (0.1, 0.05):
        JOBS.append((' '.join(f'{k} {v}' for k, v in kw.items()) + f' base {base}', dict(base=base, **kw)))
if __name__ == '__main__':
    from multiprocessing import get_context
    with get_context('fork').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        for line in pool.imap(run, JOBS):
            print(line, flush=True)
    print('RULE_REAL_DONE')
