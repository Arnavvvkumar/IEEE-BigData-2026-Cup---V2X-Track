"""Round 2 of the scan-point rule (pre-registered).
VEHICLE: on the full v16 pipeline (calibrate -> truck rule a.5 base.1 -> dedupe), vehicle-class boxes with < thr non-ground points inside
   (box grown 0.1 m) get score * m; thr in {5, 10, 20}, m in {0.5, 0.25}. Vehicle AP all / half A / half B. Trucks untouched.
PEDESTRIAN (on top of pts<5 x0.25): (a) also apply the rule to every HEDGE VARIANT using the variant's own point count;
   (b) graded: 5 <= n < 8 -> x0.7.
usage: minpts_more.py <set dir> <full norule pred name> <ped-only norule pred name>"""
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
d, full, pedname = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
clouds = {f: np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
NG = {f: calib.nonground(clouds[f]) for f in ids}


def npts(f, b, grow=0.1):
    q = NG[f]; b = np.asarray(b, float)
    near = np.hypot(q[:, 0] - b[0], q[:, 1] - b[1]) < np.hypot(b[3], b[4]) / 2 + 0.3
    r = q[near] - b[:3]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    xx = r[:, 0] * cs - r[:, 1] * sn; yy = r[:, 0] * sn + r[:, 1] * cs
    return int(((np.abs(xx) <= b[3] / 2 + grow) & (np.abs(yy) <= b[4] / 2 + grow) & (np.abs(r[:, 2]) <= b[5] / 2 + grow)).sum())


def ap(p, cls):
    gt = {f: [{'class': cls, 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == cls] for f in ids}
    out = []
    for h in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: [x for x in p[f] if x['class'] == cls] for f in h}, {f: gt[f] for f in h}, [cls],
                                     iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class'][cls]['AP_0.5_R40'], 2))
    return out


def rescore(p, cls, fn):
    return {f: [({**x, 'score': x['score'] * fn(npts(f, x['box']))} if x['class'] == cls else x) for x in p[f]] for f in ids}


# vehicles
pv = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{full}.json').read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))
pv = {f: [x for x in pv[f] if x['class'] == 'vehicle'] for f in ids}
cnt = {f: [npts(f, x['box']) for x in pv[f]] for f in ids}
print(f'{d.name} VEH base            vehicle AP {ap(pv, "vehicle")} | boxes {sum(map(len, pv.values()))} with <5 pts '
      f'{sum(c < 5 for v in cnt.values() for c in v)} <10 {sum(c < 10 for v in cnt.values() for c in v)} <20 {sum(c < 20 for v in cnt.values() for c in v)}', flush=True)
for thr in (5, 10, 20):
    for m in (0.5, 0.25):
        q = {f: [({**x, 'score': x['score'] * m} if c < thr else x) for x, c in zip(pv[f], cnt[f])] for f in ids}
        print(f'{d.name} VEH pts<{thr:<2d} x{m:<4}     vehicle AP {ap(q, "vehicle")}', flush=True)

# pedestrians
pp = json.loads((d / f'pred_{pedname}.json').read_text())
pp = {f: [x for x in pp[f] if x['class'] == 'pedestrian'] for f in ids}
r5 = calib.ped_min_points(pp, clouds)
print(f'{d.name} PED pts<5 x.25 (v17)   pedestrian AP {ap(calib.hedge_pedestrians(r5), "pedestrian")}', flush=True)
h = calib.hedge_pedestrians(r5)
nprim = {f: len(r5[f]) for f in ids}
va = {f: h[f][:nprim[f]] + [({**x, 'score': x['score'] * 0.25} if npts(f, x['box']) < 5 else x) for x in h[f][nprim[f]:]] for f in ids}
print(f'{d.name} PED (a) + variants own  pedestrian AP {ap(va, "pedestrian")}', flush=True)
vb = rescore(r5, 'pedestrian', lambda n: 0.7 if 5 <= n < 8 else 1.0)
print(f'{d.name} PED (b) graded 5-8 x.7  pedestrian AP {ap(calib.hedge_pedestrians(vb), "pedestrian")}', flush=True)
print('MINPTS_MORE_DONE')
