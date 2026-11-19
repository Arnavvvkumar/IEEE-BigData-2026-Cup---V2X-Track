"""Diagnosis: do real pedestrian label sizes follow the extent of the lidar points? (dev frames; label statistics only)
Points are gathered around our PREDICTED box (as available at test time), sizes compared with the matched label."""
import os
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
gt = json.loads((KIT / 'ground_truth.json').read_text())
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
pred = json.loads(Path(sys.argv[1]).read_text())
PLANE = np.array([-0.006, -0.1129, -2.8702])
rows = []
for f in ids:
    p = np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3).astype(np.float64)
    p = p[np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)]
    hag = p[:, 2] - (p[:, :2] @ PLANE[:2] + PLANE[2])
    p, hag = p[hag > 0.12], hag[hag > 0.12]
    peds = sorted((d for d in pred[f] if d['class'] == 'pedestrian'), key=lambda d: -d['score'])
    pb = np.array([d['box'] for d in peds]).reshape(-1, 7)
    for g in gt[f]:
        if g['class'] != 'pedestrian' or not len(pb):
            continue
        b = np.array(g['box'], float)
        near = np.flatnonzero(np.hypot(pb[:, 0] - b[0], pb[:, 1] - b[1]) < 0.6)
        if not len(near):
            continue
        q = pb[near[0]]
        m = (np.hypot(p[:, 0] - q[0], p[:, 1] - q[1]) < 0.55) & (hag < 2.2)
        if m.sum() < 5:
            continue
        pts = p[m]
        u = q[:2] / np.hypot(q[0], q[1])
        v = np.array([-u[1], u[0]])
        ev_, _ = np.linalg.eigh(np.cov(pts[:, :2].T))
        rows.append([b[3], b[4], b[5], max(b[3], b[4]), min(b[3], b[4]), np.ptp(pts[:, :2] @ u), np.ptp(pts[:, :2] @ v), hag[m].max(),
                     np.sqrt(ev_[1]), np.sqrt(max(ev_[0], 0)), m.sum(), np.hypot(b[0], b[1]), q[3], q[4], q[5],
                     b[2] + b[5] / 2 - (b[:2] @ PLANE[:2] + PLANE[2])])
a = np.array(rows)
names = ['lab_l', 'lab_w', 'lab_h', 'lab_max', 'lab_min', 'rad_ext', 'tan_ext', 'top_hag', 'pca_major', 'pca_minor', 'n', 'range', 'pred_l', 'pred_w',
         'pred_h', 'lab_top_hag']
print('n', len(a))
for t in ('lab_l', 'lab_w', 'lab_h', 'lab_max', 'lab_min', 'lab_top_hag'):
    i = names.index(t)
    print(t, ' '.join(f'{k}={np.corrcoef(a[:, i], a[:, j])[0, 1]:+.2f}' for j, k in enumerate(names) if k not in ('lab_l', 'lab_w', 'lab_h', 'lab_max', 'lab_min', 'lab_top_hag')))
for t, x in (('lab_top_hag', 'top_hag'), ('lab_h', 'top_hag'), ('lab_max', 'tan_ext'), ('lab_max', 'pca_major'), ('lab_min', 'pca_minor'), ('lab_l', 'pred_l'), ('lab_h', 'pred_h')):
    i, j = names.index(t), names.index(x)
    A = np.c_[a[:, j], np.ones(len(a))]
    coef, *_ = np.linalg.lstsq(A, a[:, i], rcond=None)
    resid = a[:, i] - A @ coef
    print(f'{t} ~ {coef[0]:.3f}*{x} + {coef[1]:.3f}   residual sd {resid.std():.3f} (label sd {a[:, i].std():.3f})')
