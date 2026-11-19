"""Codex idea: suppress duplicate pedestrian parents that claim the same scan returns (after the v17 min-points rule, before hedging).
Parents in descending score order; P_i = indices of non-ground points inside the ungrown box. A parent j is a duplicate of a RETAINED
higher-scored parent i (centre distance < 1 m) if |P_i & P_j| / |P_i | P_j| >= 0.8, both have >= 5 points, and 3D IoU(i, j) >= 0.5.
Duplicate parents (and their hedge variants) get score * m, m in {0.5, 0.25}. Hedge eligibility is frozen from v17 (pre-suppression).
usage: ped_dup_points.py <set dir> <ped-only norule pred name>"""
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
P = calib.ped_min_points({f: [x for x in pp[f] if x['class'] == 'pedestrian'] for f in ids}, clouds)


def inside(q, idx, b):
    r = q[idx] - b[:3]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    xx = r[:, 0] * cs - r[:, 1] * sn; yy = r[:, 0] * sn + r[:, 1] * cs
    return set(idx[(np.abs(xx) <= b[3] / 2) & (np.abs(yy) <= b[4] / 2) & (np.abs(r[:, 2]) <= b[5] / 2)].tolist())


DUP, ndup = {}, 0
for f in ids:
    q = calib.nonground(clouds[f])
    order = np.argsort([-x['score'] for x in P[f]])
    kept, dup = [], set()
    for k in order:
        b = np.array(P[f][k]['box'], float)
        idx = np.flatnonzero(np.hypot(q[:, 0] - b[0], q[:, 1] - b[1]) < 1.0)
        s = inside(q, idx, b)
        isdup = False
        if len(s) >= 5:
            for kb, ks in kept:
                if np.hypot(*(b[:2] - kb[:2])) < 1.0 and len(ks) >= 5 and len(s & ks) / len(s | ks) >= 0.8 and ev._iou_3d(b, kb) >= 0.5:
                    isdup = True
                    break
        if isdup:
            dup.add(int(k))
        else:
            kept.append((b, s))
    DUP[f] = dup; ndup += len(dup)
print(f'{d.name} parents {sum(map(len, P.values()))} | duplicates flagged {ndup}', flush=True)
H = calib.hedge_pedestrians(P)   # eligibility frozen from v17


def ap(p):
    out = []
    for hh in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: p[f] for f in hh}, {f: gtd[f] for f in hh}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 3))
    return out


print(f'{d.name} v17 base          ped AP {ap(H)}', flush=True)
nv = len(calib.PED_HEDGE)
for m in (0.5, 0.25):
    out = {}
    for f in ids:
        n = len(P[f]); prim = H[f][:n]; var = H[f][n:]
        # hedge_pedestrians appends, per eligible parent in order, len(PED_HEDGE) variants
        elig = [k for k, x in enumerate(P[f]) if x['score'] >= 0.05]
        newp = [({**x, 'score': x['score'] * m} if k in DUP[f] else x) for k, x in enumerate(prim)]
        newv = []
        for j, k in enumerate(elig):
            for x in var[j * nv:(j + 1) * nv]:
                newv.append({**x, 'score': x['score'] * m} if k in DUP[f] else x)
        assert len(newv) == len(var)
        out[f] = newp + newv
    print(f'{d.name} dup x{m:<4}         ped AP {ap(out)}', flush=True)
print('DUP_DONE')
