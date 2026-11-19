"""Background pedestrian FPs cost ~5.6 AP (dev150). Can the scan's own points inside a box separate them (label-free at test time)?
Features per PRIMARY pedestrian box (before hedging): n = non-ground points inside the box grown by 0.1 m, top = highest of those points
above the local ground plane, ext = their vertical extent. Prints feature quantiles for TP / localisation-FP / background-FP among the
top-5000, then the hedged pedestrian AP (all / half A / half B) for a pre-registered set of score rules:
    score *= m  when the feature is below a threshold  (m in {0.5, 0.25})
usage: ped_pts_rank.py <set dir> <norule pred name> [rules: all | 'name,...']"""
import os
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
gt = {f: [np.array(g['box'], float) for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
gtd = {f: [{'class': 'pedestrian', 'box': g.tolist()} for g in gt[f]] for f in ids}
pred = json.loads((d / f'pred_{name}.json').read_text())
P = {f: [x for x in pred[f] if x['class'] == 'pedestrian'] for f in ids}


def feats(f, dets):
    q = calib.nonground(np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3))
    hz = q[:, 2] - (q[:, :2] @ calib.PLANE[:2] + calib.PLANE[2])
    out = []
    for x in dets:
        b = np.array(x['box'], float)
        near = np.hypot(q[:, 0] - b[0], q[:, 1] - b[1]) < 1.2
        r = q[near] - b[:3]
        cs, sn = np.cos(-b[6]), np.sin(-b[6])
        xx = r[:, 0] * cs - r[:, 1] * sn; yy = r[:, 0] * sn + r[:, 1] * cs
        m = (np.abs(xx) <= b[3] / 2 + 0.1) & (np.abs(yy) <= b[4] / 2 + 0.1) & (np.abs(r[:, 2]) <= b[5] / 2 + 0.1)
        h = hz[near][m]
        out.append((int(m.sum()), float(h.max()) if len(h) else 0.0, float(h.max() - h.min()) if len(h) else 0.0))
    return out


F = {f: feats(f, P[f]) for f in ids}


def ap(p):
    out = []
    for hh in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: p[f] for f in hh}, {f: gtd[f] for f in hh}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 2))
    return out


# feature quantiles by type (top-5000 primaries, greedy matching on primaries only)
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
        rows.append((P[f][k]['score'], t, F[f][k]))
rows.sort(key=lambda r: -r[0]); rows = rows[:5000]
for t in ('TP', 'loc', 'bg'):
    v = np.array([r[2] for r in rows if r[1] == t])
    if len(v) == 0: continue
    print(f'{d.name} {t:3s} n={len(v):4d} | pts q05/q25/q50 {np.percentile(v[:, 0], [5, 25, 50]).round(0).tolist()} | '
          f'top q05/q25/q50 {np.percentile(v[:, 1], [5, 25, 50]).round(2).tolist()} | ext q05/q25/q50 {np.percentile(v[:, 2], [5, 25, 50]).round(2).tolist()}', flush=True)

RULES = {'base': None}
for thr in (5, 10):
    for m in (0.5, 0.25):
        RULES[f'pts<{thr} x{m}'] = (0, thr, m)
for thr in (0.8, 1.0):
    for m in (0.5, 0.25):
        RULES[f'top<{thr} x{m}'] = (1, thr, m)
for thr in (0.5, 0.8):
    for m in (0.5, 0.25):
        RULES[f'ext<{thr} x{m}'] = (2, thr, m)
sel = list(RULES) if len(sys.argv) < 4 or sys.argv[3] == 'all' else ['base'] + sys.argv[3].split(',')
for rn in sel:
    rule = RULES[rn]
    q = {}
    for f in ids:
        q[f] = []
        for x, ft in zip(P[f], F[f]):
            s = x['score']
            if rule and ft[rule[0]] < rule[1]:
                s *= rule[2]
            q[f].append({**x, 'score': s})
    print(f'{d.name} {rn:16s} ped AP {ap(calib.hedge_pedestrians(q))}', flush=True)
