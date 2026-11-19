"""Pedestrian near-miss geometry on the real dev frames: is the centre error radial (towards the sensor), and how do the
real pedestrian boxes relate to the points inside them? Statistics only. usage: ped_offsets.py <predictions.json>"""
import os
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
gt = json.loads((KIT / 'ground_truth.json').read_text())
pred = json.loads(Path(sys.argv[1]).read_text())
rad, tan, dz, dl, dw, dh, sc = [], [], [], [], [], [], []
cen_r, cen_t, top_gap, bot_gap, fill = [], [], [], [], []
for f, gts in gt.items():
    pts = np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3)
    P = [p for p in pred[f] if p['class'] == 'pedestrian' and p['score'] >= 0.15]
    PB = np.array([p['box'] for p in P]).reshape(-1, 7)
    for g in gts:
        if g['class'] != 'pedestrian':
            continue
        b = np.array(g['box'])
        u = b[:2] / np.linalg.norm(b[:2])
        t = np.array([-u[1], u[0]])
        m = (np.abs(pts[:, 0] - b[0]) < 0.6) & (np.abs(pts[:, 1] - b[1]) < 0.6) & (np.abs(pts[:, 2] - b[2]) < b[5] / 2 + 0.3)
        if m.sum() >= 8:
            q = pts[m]
            c = q[:, :2].mean(0) - b[:2]
            cen_r.append(c @ u); cen_t.append(c @ t)
            top_gap.append(b[2] + b[5] / 2 - q[:, 2].max()); bot_gap.append(q[:, 2].min() - (b[2] - b[5] / 2))
            fill.append([np.ptp(q[:, 0] * u[0] + q[:, 1] * u[1]), np.ptp(q[:, 0] * t[0] + q[:, 1] * t[1])])
        if len(PB):
            d = np.hypot(PB[:, 0] - b[0], PB[:, 1] - b[1])
            i = int(d.argmin())
            if d[i] < 1.0:
                e = PB[i, :2] - b[:2]
                rad.append(e @ u); tan.append(e @ t); dz.append(PB[i, 2] - b[2])
                dl.append(PB[i, 3] - b[3]); dw.append(PB[i, 4] - b[4]); dh.append(PB[i, 5] - b[5]); sc.append(P[i]['score'])


def q(v):
    return np.percentile(v, [10, 50, 90]).round(3).tolist()


print('matched within 1 m:', len(rad))
print('pred-gt radial (+ = away from sensor) p10/50/90', q(rad), ' tangential', q(tan), ' |radial| median', round(float(np.median(np.abs(rad))), 3),
      '|tangential| median', round(float(np.median(np.abs(tan))), 3))
print('dz', q(dz), 'dl', q(dl), 'dw', q(dw), 'dh', q(dh), 'score', q(sc))
print('REAL boxes vs their own points (n=%d): point-centroid minus box centre radial' % len(cen_r), q(cen_r), 'tangential', q(cen_t))
print('gap box top - highest point', q(top_gap), ' lowest point - box bottom', q(bot_gap), ' point extent radial/tangential median', np.median(fill, 0).round(2).tolist())
