"""Truck gap step 3: for each Truck/Van/Bus GT (and non-static ones), what do we output there? best truck pred (IoU, score),
best vehicle pred (IoU, score, size). Tells whether the miss is class confusion, box size, or no detection at all.
usage: truck_gap_miss.py <set dir> <norule pred name>"""
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
ids = sorted(G)
raw = json.loads((d / f'pred_{name}.json').read_text())
pred = calib.truck_dedupe(calib.truck_rule(raw, **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))
print('truck rule params', {**calib.TRUCK, 'a': 0.5, 'base': 0.1})


def best(g, dets, cls):
    b = np.array(g['box']); out = (0.0, 0.0, None)
    for x in dets:
        if x['class'] != cls:
            continue
        if np.hypot(x['box'][0] - b[0], x['box'][1] - b[1]) > 6:
            continue
        i = ev._iou_3d(b, np.array(x['box'])); bi = ev._bev_iou(b, np.array(x['box']))
        if (i, bi) > out[:2]:
            out = (i, bi, x)
    return out


rows = defaultdict(list)
for f in ids:
    for g in G[f]:
        if g['class'] != 'truck' or g['type'] == 'TrashCan':
            continue
        it, bt, xt = best(g, pred[f], 'truck')
        iv, bv, xv = best(g, pred[f], 'vehicle')
        rows[g['type']].append((g, it, bt, xt, iv, bv, xv))
for t, R in rows.items():
    n = len(R)
    tr_ok = sum(r[1] >= 0.5 for r in R)
    tr_bev = sum(r[2] >= 0.5 and r[1] < 0.5 for r in R)
    veh_ok = sum(r[4] >= 0.5 for r in R)
    veh_bev = sum(r[5] >= 0.3 for r in R)
    none = sum(r[2] < 0.1 and r[5] < 0.1 for r in R)
    print(f'\n{d.name} {t}: n={n} | truck IoU>=.5 {tr_ok} | truck BEV>=.5 but 3D<.5 {tr_bev} | vehicle IoU>=.5 {veh_ok} | vehicle BEV>=.3 {veh_bev} | nothing near {none}')
    gs = np.array([r[0]['box'][3:6] for r in R]); print(f'  GT size L/W/H median {np.median(gs, 0).round(2)}')
    vs = [r for r in R if r[6] is not None and r[5] >= 0.3]
    if vs:
        vb = np.array([r[6]['box'][3:6] for r in vs]); sc = np.array([r[6]['score'] for r in vs])
        print(f'  matched vehicle preds: size median {np.median(vb, 0).round(2)}, score median {np.median(sc):.3f} [p25 {np.percentile(sc, 25):.3f}, p75 {np.percentile(sc, 75):.3f}]'
              f', 3D IoU median {np.median([r[4] for r in vs]):.2f}')
        # would a truck copy of that vehicle box, resized to the GT-type median size, hit IoU 0.5?
        hit_resize = 0
        for r in vs:
            b = np.array(r[6]['box'], float); b[3:6] = np.median(gs, 0); b[2] = r[6]['box'][2] - r[6]['box'][5] / 2 + b[5] / 2
            hit_resize += ev._iou_3d(np.array(r[0]['box']), b) >= 0.5
        print(f'  truck copy of the vehicle box at median {t} size would match IoU>=.5: {hit_resize}/{len(vs)}')
    ts = [r for r in R if r[3] is not None and r[2] >= 0.3]
    if ts:
        tb = np.array([r[3]['box'][3:6] for r in ts]); sc = np.array([r[3]['score'] for r in ts])
        print(f'  matched truck preds:   size median {np.median(tb, 0).round(2)}, score median {np.median(sc):.3f}, 3D IoU median {np.median([r[1] for r in ts]):.2f}')
