"""Round 3, pedestrian only, on cached v16-base predictions (fused + static down-weight + calibration, before hedging).
All variants start from v17 (calib.ped_min_points: < 5 points -> x0.25), then hedge (calib.hedge_pedestrians), full-precision AP.
E1 density-normalised point count: n * (r/20)^2 < T  ->  score x m. (Beam geometry: points on an object fall as 1/r^2, so a near box with
   few points is more suspicious than a far one; a hard 5-point rule treats both the same.)  T in {8, 16}, m in {0.5, 0.25}.
E2 primary size scale: l and w of every primary box x s (hedge variants are built from the scaled box). Real GT median 0.65 x 0.67,
   predicted 0.76 x 0.74.  s in {0.95, 0.9, 0.85}.
usage: ped_density_size.py <set dir> <ped-only norule pred name> [workers]"""
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


def npts(q, b, grow=0.1):
    near = np.hypot(q[:, 0] - b[0], q[:, 1] - b[1]) < 1.2
    r = q[near] - b[:3]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    xx = r[:, 0] * cs - r[:, 1] * sn; yy = r[:, 0] * sn + r[:, 1] * cs
    return int(((np.abs(xx) <= b[3] / 2 + grow) & (np.abs(yy) <= b[4] / 2 + grow) & (np.abs(r[:, 2]) <= b[5] / 2 + grow)).sum())


N = {}
for f in ids:
    q = calib.nonground(clouds[f])
    N[f] = [npts(q, np.array(x['box'], float)) for x in P0[f]]
P1 = {f: [({**x, 'score': x['score'] * 0.25} if n < 5 else x) for x, n in zip(P0[f], N[f])] for f in ids}


def ap(p):
    out = []
    for hh in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: p[f] for f in hh}, {f: gtd[f] for f in hh}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 3))
    return out


def run(job):
    label, kind, par = job
    if kind == 'base':
        P = P1
    elif kind == 'dens':
        T, m = par
        P = {f: [({**x, 'score': x['score'] * m} if n * (np.hypot(x['box'][0], x['box'][1]) / 20) ** 2 < T else x) for x, n in zip(P1[f], N[f])] for f in ids}
    else:
        P = {f: [{**x, 'box': x['box'][:3] + [x['box'][3] * par, x['box'][4] * par] + x['box'][5:]} for x in P1[f]] for f in ids}
    return f'{d.name} {label:22s} ped AP {ap(calib.hedge_pedestrians(P))}'


JOBS = [('v17 base', 'base', None), ('dens<8 x0.5', 'dens', (8, 0.5)), ('dens<16 x0.5', 'dens', (16, 0.5)), ('dens<8 x0.25', 'dens', (8, 0.25)),
        ('size x0.95', 'size', 0.95), ('size x0.90', 'size', 0.90), ('size x0.85', 'size', 0.85)]
if __name__ == '__main__':
    r = np.array([np.hypot(x['box'][0], x['box'][1]) for f in ids for x in P1[f]])
    n = np.array([k for f in ids for k in N[f]])
    print(f'{d.name} boxes {len(n)} | n*(r/20)^2 quantiles q10/q25/q50/q75 {np.percentile(n * (r / 20) ** 2, [10, 25, 50, 75]).round(1).tolist()}', flush=True)
    from multiprocessing import get_context
    with get_context('fork').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 3) as pool:
        for line in pool.imap(run, JOBS):
            print(line, flush=True)
    print('DENSITY_SIZE_DONE')
