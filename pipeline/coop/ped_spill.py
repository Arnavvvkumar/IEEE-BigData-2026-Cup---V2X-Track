"""Localisation FPs (IoU 0.1-0.5) cost up to ~14 AP. Do the scan points tell a well-placed box from a misplaced one (label-free)?
Features per primary pedestrian box (after the v17 min-points rule): spill = 1 - n(box+0.1 m) / n(box+0.35 m) (points just outside the
box), off = |point-centroid - box centre| in BEV / box half-diagonal, fill = n(box) / n(box+0.1). Quantiles for TP vs loc-FP vs bg-FP
among the top-5000; then pre-registered rules: score *= m when spill > t (t in {0.3, 0.5}) or off > t (t in {0.5, 0.8}), m in {0.7, 0.5}.
usage: ped_spill.py <set dir> <ped-only norule pred name>"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
gt = {f: [np.array(g['box'], float) for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
gtd = {f: [{'class': 'pedestrian', 'box': g.tolist()} for g in gt[f]] for f in ids}
clouds = {f: np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
pp = json.loads((d / f'pred_{name}.json').read_text())
P = calib.ped_min_points({f: [x for x in pp[f] if x['class'] == 'pedestrian'] for f in ids}, clouds)


def feats(f, b):
    q = NG[f]; b = np.asarray(b, float)
    near = np.hypot(q[:, 0] - b[0], q[:, 1] - b[1]) < 1.5
    r = q[near] - b[:3]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    xx = r[:, 0] * cs - r[:, 1] * sn; yy = r[:, 0] * sn + r[:, 1] * cs; zz = r[:, 2]
    inn = lambda g: (np.abs(xx) <= b[3] / 2 + g) & (np.abs(yy) <= b[4] / 2 + g) & (np.abs(zz) <= b[5] / 2 + 0.1)
    n0, n1, n3 = int(inn(0.0).sum()), int(inn(0.1).sum()), int(inn(0.35).sum())
    m = inn(0.1)
    off = float(np.hypot(xx[m].mean(), yy[m].mean()) / (np.hypot(b[3], b[4]) / 2)) if m.sum() else 1.0
    return (1 - n1 / n3) if n3 else 1.0, off, (n0 / n1) if n1 else 0.0


NG = {f: calib.nonground(clouds[f]) for f in ids}
FT = {f: [feats(f, x['box']) for x in P[f]] for f in ids}
rows = []
for f in ids:
    used = [False] * len(gt[f])
    for k in np.argsort([-x['score'] for x in P[f]]):
        b = np.array(P[f][k]['box'], float)
        ious = [ev._iou_3d(b, g) for g in gt[f]]
        bi, bj = 0.0, -1
        for j, i in enumerate(ious):
            if not used[j] and i > bi: bi, bj = i, j
        t = 'TP' if bi >= 0.5 else ('loc' if (max(ious) if ious else 0) >= 0.1 else 'bg')
        if t == 'TP': used[bj] = True
        rows.append((P[f][k]['score'], t, FT[f][k]))
rows.sort(key=lambda r: -r[0]); rows = rows[:5000]
for t in ('TP', 'loc', 'bg'):
    v = np.array([r[2] for r in rows if r[1] == t])
    print(f'{d.name} {t:3s} n={len(v):4d} | spill q50/q75/q90 {np.percentile(v[:, 0], [50, 75, 90]).round(2).tolist()} | '
          f'off q50/q75/q90 {np.percentile(v[:, 1], [50, 75, 90]).round(2).tolist()} | fill q10/q25/q50 {np.percentile(v[:, 2], [10, 25, 50]).round(2).tolist()}', flush=True)


def ap(p):
    out = []
    for hh in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: p[f] for f in hh}, {f: gtd[f] for f in hh}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 2))
    return out


print(f'{d.name} v17 base           ped AP {ap(calib.hedge_pedestrians(P))}', flush=True)
for fi, nm, ts in ((0, 'spill', (0.3, 0.5)), (1, 'off', (0.5, 0.8))):
    for t in ts:
        for m in (0.7, 0.5):
            q = {f: [({**x, 'score': x['score'] * m} if ft[fi] > t else x) for x, ft in zip(P[f], FT[f])] for f in ids}
            print(f'{d.name} {nm}>{t} x{m:<4}     ped AP {ap(calib.hedge_pedestrians(q))}', flush=True)
print('SPILL_DONE')
