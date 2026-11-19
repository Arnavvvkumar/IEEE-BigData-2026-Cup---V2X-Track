"""Read-only diagnostic: why do detections fail on the 50 public real dev frames? (centre offsets of nearest predictions)"""
import os
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
gt = json.loads((W / 'experiments/real_reference_audit/real_dev_cw1/ground_truth.json').read_text())
pred = json.loads((W / sys.argv[1]).read_text())
for c in ('vehicle', 'pedestrian', 'truck'):
    dz, dxy, dl, found, n = [], [], [], 0, 0
    gsz = []
    for f, gts in gt.items():
        ps = [p for p in pred.get(f, []) if p['class'] == c and p['score'] >= 0.1]
        P = np.array([p['box'] for p in ps]).reshape(-1, 7)
        for g in gts:
            if g['class'] != c:
                continue
            n += 1
            b = np.array(g['box'])
            gsz.append(b[3:6])
            if not len(P):
                continue
            d = np.hypot(P[:, 0] - b[0], P[:, 1] - b[1])
            i = int(d.argmin())
            if d[i] < 2.0:
                found += 1
                dxy.append(d[i]); dz.append(P[i, 2] - b[2]); dl.append(P[i, 3:6] - b[3:6])
    print(c, 'gt', n, 'gt median lwh', np.median(gsz, 0).round(2) if gsz else None, 'gt median z', None,
          '| pred within 2 m BEV:', found, '| median dxy', round(float(np.median(dxy)), 2) if dxy else None,
          'median dz(pred-gt)', round(float(np.median(dz)), 2) if dz else None,
          'median dlwh', np.median(dl, 0).round(2) if dl else None)
zs = [g['box'][2] - g['box'][5] / 2 for gts in gt.values() for g in gts]
print('real GT box bottom z median', round(float(np.median(zs)), 2))
