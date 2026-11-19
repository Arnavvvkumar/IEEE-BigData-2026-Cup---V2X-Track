"""Can predicted box size separate real Truck/Van labels from Car labels? For every label of type Car / Truck / Van
(>=5 points), take the best-overlapping vehicle-or-truck prediction (IoU >= 0.5) and list its predicted class, score and size.
usage: truck_sizes.py <set dir>"""
import os
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d = Path(sys.argv[1])
gt = json.loads((d / 'gt_coop.json').read_text())
pred = json.loads((d / 'pred_cw_d_calibrated.json').read_text())
rows = {'Car': [], 'Truck': [], 'Van': []}
for f, v in gt.items():
    P = [x for x in pred[f] if x['class'] in ('vehicle', 'truck') and x['score'] >= 0.03]
    for g in v:
        if g['type'] not in rows or g['points'] < 5:
            continue
        b = np.array(g['box'])
        cand = [(ev._iou_3d(np.array(x['box']), b), x) for x in P if np.hypot(x['box'][0] - b[0], x['box'][1] - b[1]) < 2.5]
        vb = max((c for c in cand if c[1]['class'] == 'vehicle'), key=lambda c: c[0], default=(0, None))
        tb = max((c for c in cand if c[1]['class'] == 'truck'), key=lambda c: c[0], default=(0, None))
        rows[g['type']].append((vb[0], vb[1]['score'] if vb[1] else 0, *(vb[1]['box'][3:6] if vb[1] else (0, 0, 0)),
                                tb[0], tb[1]['score'] if tb[1] else 0, *b[3:6], g['points']))
for t, r in rows.items():
    a = np.array(r)
    hv, ht = a[:, 0] >= 0.5, a[:, 5] >= 0.5
    print(f'{t}: n={len(a)} | vehicle-pred hit {hv.mean():.2f} (median score {np.median(a[hv, 1]) if hv.any() else 0:.2f}) | truck-pred hit {ht.mean():.2f} (median score {np.median(a[ht, 6]) if ht.any() else 0:.2f})')
    if hv.any():
        print('   size of the matching vehicle prediction l,w,h: median', np.median(a[hv, 2:5], 0).round(2).tolist(), '| p10', np.percentile(a[hv, 2:5], 10, 0).round(2).tolist(),
              '| p90', np.percentile(a[hv, 2:5], 90, 0).round(2).tolist())
    print('   label l,w,h median', np.median(a[:, 7:10], 0).round(2).tolist())
car, trk = np.array(rows['Car']), np.array(rows['Truck'] + rows['Van'])
car, trk = car[car[:, 0] >= 0.5], trk[trk[:, 0] >= 0.5]
print('share of matched vehicle predictions above a size threshold (Car labels vs Truck/Van labels):')
for name, col, thrs in (('length', 2, (5.0, 5.2, 5.4, 5.6)), ('width', 3, (2.1, 2.15, 2.2, 2.3)), ('height', 4, (1.8, 1.9, 2.0, 2.1))):
    print('  ', name, ' '.join(f'>={t}: car {np.mean(car[:, col] >= t):.2f} / truck {np.mean(trk[:, col] >= t):.2f}' for t in thrs))
