"""Round 4, pedestrian only (v16-base cached predictions -> v17-style rule -> hedging; full-precision AP).
C: which point count best mirrors the label rule (>= 5 sensor points in the GT box)? v17 counts NON-GROUND points (> 0.12 m above the
   plane) in the box grown 0.1 m. The label count includes the points near the ground (feet, lower legs). Variants (thr 5, x0.25):
   C1 non-ground, grow 0 | C2 all points, grow 0 | C3 all points, grow 0.1 | (v17 = non-ground, grow 0.1 is the base).
S: primary box size scale UP: x1.03, x1.05, x1.10 (down-scaling x0.95 / 0.90 lost 0.3 / 3.1 AP on dev150, so check the other side).
usage: ped_count_defs.py <set dir> <ped-only norule pred name> [workers]"""
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
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
gtd = {f: [{'class': 'pedestrian', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
clouds = {f: np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
pp = json.loads((d / f'pred_{name}.json').read_text())
P0 = {f: [x for x in pp[f] if x['class'] == 'pedestrian'] for f in ids}


def allpts(p):
    p = p.astype(np.float64)
    return p[np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)]


def npts(q, b, grow):
    near = np.hypot(q[:, 0] - b[0], q[:, 1] - b[1]) < 1.2
    r = q[near] - b[:3]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    xx = r[:, 0] * cs - r[:, 1] * sn; yy = r[:, 0] * sn + r[:, 1] * cs
    return int(((np.abs(xx) <= b[3] / 2 + grow) & (np.abs(yy) <= b[4] / 2 + grow) & (np.abs(r[:, 2]) <= b[5] / 2 + grow)).sum())


DEFS = {'base': ('ng', 0.1), 'C1': ('ng', 0.0), 'C2': ('all', 0.0), 'C3': ('all', 0.1)}
N = {k: {} for k in DEFS}
for f in ids:
    qng, qall = calib.nonground(clouds[f]), allpts(clouds[f])
    for k, (kind, g) in DEFS.items():
        N[k][f] = [npts(qng if kind == 'ng' else qall, np.array(x['box'], float), g) for x in P0[f]]


def ap(p):
    out = []
    for hh in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: p[f] for f in hh}, {f: gtd[f] for f in hh}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 3))
    return out


def run(job):
    label, kind, par = job
    key, s = (par, 1.0) if kind == 'count' else ('base', par)
    P = {f: [({**x, 'score': x['score'] * 0.25} if n < 5 else x) for x, n in zip(P0[f], N[key][f])] for f in ids}
    if s != 1.0:
        P = {f: [{**x, 'box': x['box'][:3] + [x['box'][3] * s, x['box'][4] * s] + x['box'][5:]} for x in P[f]] for f in ids}
    return f'{d.name} {label:18s} ped AP {ap(calib.hedge_pedestrians(P))}'


JOBS = [('C1 ng grow0', 'count', 'C1'), ('C2 all grow0', 'count', 'C2'), ('C3 all grow.1', 'count', 'C3'),
        ('size x1.03', 'size', 1.03), ('size x1.05', 'size', 1.05), ('size x1.10', 'size', 1.10)]
if __name__ == '__main__':
    for k in DEFS:
        n = np.array([v for f in ids for v in N[k][f]])
        print(f'{d.name} count {k:5s}: share < 5 = {(n < 5).mean():.3f}, median {np.median(n):.0f}', flush=True)
    from multiprocessing import get_context
    with get_context('fork').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 2) as pool:
        for line in pool.imap(run, JOBS):
            print(line, flush=True)
    print('COUNT_DEFS_DONE')
