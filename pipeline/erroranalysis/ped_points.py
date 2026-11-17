"""Diagnosis: how are real pedestrian labels placed relative to the lidar points, and how do our boxes differ? (dev frames)
 A. label vs points: for each labelled pedestrian, the non-ground points inside the label (grown 0.1 m): count, point extent vs
    label size along the label axes, offset of the point-extent midpoint / centroid from the label centre (radial, tangential, z).
 B. prediction vs label: for each label, the best-scoring pedestrian prediction within 0.6 m: offset (radial, tangential, z),
    size difference; medians and robust spreads, also by range band.
usage: ped_points.py <predictions.json>"""
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
gt = json.loads((KIT / 'ground_truth.json').read_text())
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
pred = json.loads(Path(sys.argv[1]).read_text())
PLANE = np.array([-0.006, -0.1129, -2.8702])


def mad(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def summ(name, x):
    x = np.asarray(x, float)
    print(f'  {name:28s} n={len(x):4d} median={np.median(x):+.3f} spread={mad(x):.3f} p10={np.percentile(x, 10):+.3f} p90={np.percentile(x, 90):+.3f}')


A, B = [], []
for f in ids:
    p = np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3).astype(np.float64)
    p = p[np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)]
    p = p[p[:, 2] - (p[:, :2] @ PLANE[:2] + PLANE[2]) > 0.12]
    peds = sorted((d for d in pred[f] if d['class'] == 'pedestrian'), key=lambda d: -d['score'])
    pb = np.array([d['box'] for d in peds]).reshape(-1, 7)
    for g in gt[f]:
        if g['class'] != 'pedestrian':
            continue
        b = np.array(g['box'], float)
        rng = np.hypot(b[0], b[1])
        u = b[:2] / rng                      # radial unit vector (away from sensor)
        v = np.array([-u[1], u[0]])
        q = p[np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) < 1.2] - b[:3]
        cs, sn = np.cos(-b[6]), np.sin(-b[6])
        x = q[:, 0] * cs - q[:, 1] * sn
        y = q[:, 0] * sn + q[:, 1] * cs
        m = (np.abs(x) <= b[3] / 2 + 0.1) & (np.abs(y) <= b[4] / 2 + 0.1) & (np.abs(q[:, 2]) <= b[5] / 2 + 0.1)
        n = int(m.sum())
        row = {'range': rng, 'n': n}
        if n >= 5:
            qq = q[m]
            mid = (qq.max(0) + qq.min(0)) / 2
            cen = qq.mean(0)
            row.update(ext_l=np.ptp(x[m]) / b[3], ext_w=np.ptp(y[m]) / b[4], ext_h=np.ptp(qq[:, 2]) / b[5],
                       mid_rad=mid[:2] @ u, mid_tan=mid[:2] @ v, mid_z=mid[2], cen_rad=cen[:2] @ u, cen_tan=cen[:2] @ v,
                       top=qq[:, 2].max() - b[5] / 2, near_rad=(qq[:, :2] @ u).min() + 0.0,
                       rad_ext=np.ptp(qq[:, :2] @ u), tan_ext=np.ptp(qq[:, :2] @ v))
        A.append(row)
        if len(pb):
            dist = np.hypot(pb[:, 0] - b[0], pb[:, 1] - b[1])
            near = np.flatnonzero(dist < 0.6)
            if len(near):
                j = near[0]                  # highest score among nearby predictions
                d = pb[j, :3] - b[:3]
                B.append({'range': rng, 'rad': d[:2] @ u, 'tan': d[:2] @ v, 'z': d[2], 'dl': pb[j, 3] - b[3], 'dw': pb[j, 4] - b[4],
                          'dh': pb[j, 5] - b[5], 'score': peds[j]['score'], 'n': n,
                          'bottom': (pb[j, 2] - pb[j, 5] / 2) - (b[2] - b[5] / 2), 'top': (pb[j, 2] + pb[j, 5] / 2) - (b[2] + b[5] / 2)})
print('A. labels vs points: labelled pedestrians', len(A))
summ('points in label', [r['n'] for r in A])
have = [r for r in A if 'ext_l' in r]
for k in ('ext_l', 'ext_w', 'ext_h', 'mid_rad', 'mid_tan', 'mid_z', 'cen_rad', 'cen_tan', 'top', 'near_rad', 'rad_ext', 'tan_ext'):
    summ(k, [r[k] for r in have])
print('B. predictions vs labels: matched', len(B), 'of', len(A))
for k in ('rad', 'tan', 'z', 'dl', 'dw', 'dh', 'bottom', 'top'):
    summ(k, [r[k] for r in B])
for lo, hi in ((0, 15), (15, 25), (25, 40), (40, 80)):
    s = [r for r in B if lo <= r['range'] < hi]
    if len(s) > 10:
        print(f' range {lo}-{hi} m: n={len(s)}')
        for k in ('rad', 'tan', 'z'):
            summ(k, [r[k] for r in s])
