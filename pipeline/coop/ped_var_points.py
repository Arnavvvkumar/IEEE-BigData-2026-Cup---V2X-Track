"""Use the scan points to choose AMONG existing hedge candidates (not to resize boxes directly, which failed before).
Labels are tighter than our boxes (GT median 0.65 x 0.67 vs predicted 0.76 x 0.74). For each primary pedestrian box (v17: after the
min-points rule) and each PED_HEDGE variant: cover_v = points inside the variant / points inside the primary grown by 0.15 m.
Pre-registered variants:
  A1/A2: variant factor k' = min(k * (1 + alpha * (cover_v - cover_0)) , 0.9) with alpha 2 / 4  (cover_0 = the primary's cover)
  S1/S2: swap - if the 0.8x0.8 variant keeps >= 95% / >= 90% of the primary's points, it takes the primary's score and the primary
         becomes a variant at 0.2x.
usage: ped_var_points.py <set dir> <ped-only norule pred name>"""
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
gtd = {f: [{'class': 'pedestrian', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
clouds = {f: np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
NG = {f: calib.nonground(clouds[f]) for f in ids}
pp = json.loads((d / f'pred_{name}.json').read_text())
P = calib.ped_min_points({f: [x for x in pp[f] if x['class'] == 'pedestrian'] for f in ids}, clouds)
H = calib.PED_HEDGE


def count(q, b, grow=0.0):
    r = q - b[:3]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    xx = r[:, 0] * cs - r[:, 1] * sn; yy = r[:, 0] * sn + r[:, 1] * cs
    return int(((np.abs(xx) <= b[3] / 2 + grow) & (np.abs(yy) <= b[4] / 2 + grow) & (np.abs(r[:, 2]) <= b[5] / 2 + 0.05)).sum())


def variants(b):
    return [np.array([b[0] + dx, b[1] + dy, b[2], b[3] * sl, b[4] * sw, b[5], b[6]]) for sl, sw, dx, dy, k in H]


# precompute covers
COV = {}
for f in ids:
    COV[f] = []
    for x in P[f]:
        b = np.array(x['box'], float)
        q = NG[f][np.hypot(NG[f][:, 0] - b[0], NG[f][:, 1] - b[1]) < 1.5]
        ref = max(count(q, b, 0.15), 1)
        COV[f].append((count(q, b) / ref, [count(q, v) / ref for v in variants(b)]))


def build(mode, par):
    out = {}
    for f in ids:
        new = []
        for x, (c0, cv) in zip(P[f], COV[f]):
            b = np.array(x['box'], float); s = x['score']
            if s < 0.05:
                new.append(x); continue
            vs = variants(b)
            prim, ps = b, s
            items = []
            if mode == 'S' and cv[0] >= par * c0 and c0 > 0:   # H[0] is the 0.8 x 0.8 variant
                prim = vs[0]
                items.append((b, s * 0.2))
                for (sl, sw, dx, dy, k), v in zip(H[1:], vs[1:]):
                    items.append((v, s * k))
            else:
                for (sl, sw, dx, dy, k), v, c in zip(H, vs, cv):
                    kk = min(k * (1 + par * (c - c0)), 0.9) if mode == 'A' else k
                    items.append((v, s * max(kk, 0.0)))
            new.append({**x, 'box': prim.tolist(), 'score': ps})
            new += [{**x, 'box': v.tolist(), 'score': sc} for v, sc in items]
        out[f] = new
    return out


def ap(p):
    out = []
    for hh in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: p[f] for f in hh}, {f: gtd[f] for f in hh}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 2))
    return out


print(f'{d.name} v17 (calib hedge)   ped AP {ap(calib.hedge_pedestrians(P))}', flush=True)
print(f'{d.name} v17 (re-implemented) ped AP {ap(build("none", 0))}', flush=True)
for lab, mode, par in (('A1 alpha 2', 'A', 2.0), ('A2 alpha 4', 'A', 4.0), ('S1 swap >=95%', 'S', 0.95), ('S2 swap >=90%', 'S', 0.90)):
    print(f'{d.name} {lab:20s} ped AP {ap(build(mode, par))}', flush=True)
print('VARPTS_DONE')
