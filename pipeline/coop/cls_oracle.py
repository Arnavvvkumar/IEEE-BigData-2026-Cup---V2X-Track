"""Pedestrian error decomposition on a staged validation set (diagnostic only; nothing here is submitted).
For every predicted pedestrian box with a labelled pedestrian within 1 m (BEV), replace ONE component of the box by the label's
(size / z+height / xy centre / yaw / all) and re-score. The AP gain of each oracle says which part of the box costs the AP.
Also: recall ceiling, offsets of matched boxes, label vs predicted size statistics, breakdown by range and point count.
usage: cls_oracle.py <set dir> <pred json name> <class>"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d = Path(sys.argv[1])
CLS = sys.argv[3]
MD = 1.0 if CLS == 'pedestrian' else 3.0
pred = json.loads((d / sys.argv[2]).read_text())
gt = {f: [g for g in v if g['points'] >= 5 and g['class'] == CLS] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
ids = sorted(gt)
P = {f: [x for x in pred[f] if x['class'] == CLS] for f in ids}


def ap(p):
    m = ev.evaluate_3d_detection(p, {f: [{'class': CLS, 'box': g['box']} for g in gt[f]] for f in ids}, [CLS], iou_thresholds=(0.5,), iou_type='3d')
    return round(100 * m['per_class'][CLS]['AP_0.5_R40'], 1)


def oracle(parts):
    out = {}
    for f in ids:
        G = np.array([g['box'] for g in gt[f]]).reshape(-1, 7)
        o = []
        for x in P[f]:
            b = np.array(x['box'], float)
            if len(G):
                dd = np.hypot(G[:, 0] - b[0], G[:, 1] - b[1])
                j = int(dd.argmin())
                if dd[j] < MD:
                    g = G[j]
                    if 'size' in parts:
                        b[3:5] = g[3:5]
                    if 'zh' in parts:
                        b[2], b[5] = g[2], g[5]
                    if 'xy' in parts:
                        b[:2] = g[:2]
                    if 'yaw' in parts:
                        b[6] = g[6]
            o.append({'box': b.tolist(), 'score': x['score'], 'class': CLS})
        out[f] = o
    return out


print('frames', len(ids), 'labelled pedestrians', sum(len(v) for v in gt.values()), 'predicted', sum(len(v) for v in P.values()))
print('AP now                 ', ap(P))
for name, parts in [('oracle l,w', {'size'}), ('oracle z,h', {'zh'}), ('oracle x,y', {'xy'}), ('oracle yaw', {'yaw'}), ('oracle l,w + yaw', {'size', 'yaw'}),
                    ('oracle x,y + z,h', {'xy', 'zh'}), ('oracle everything', {'size', 'zh', 'xy', 'yaw'})]:
    print(f'{name:23s}', ap(oracle(parts)))
for thr in (0.3, 0.1, 0.05):
    print(f'AP keeping score >= {thr}', ap({f: [x for x in P[f] if x['score'] >= thr] for f in ids}))

rows = []
for f in ids:
    B = [(np.array(x['box'], float), x['score']) for x in P[f]]
    for g in gt[f]:
        b = np.array(g['box'], float)
        best = (0.0, None, 0.0)
        for pb, s in B:
            if np.hypot(pb[0] - b[0], pb[1] - b[1]) < MD:
                i = ev._iou_3d(pb, b)
                if i > best[0]:
                    best = (i, pb, s)
        rows.append((b, best, np.hypot(b[0], b[1]), g['points'], g.get('type', '')))
iou = np.array([r[1][0] for r in rows])
print('labels: IoU>=0.5', round(float((iou >= 0.5).mean()), 3), '| 0.25-0.5', round(float(((iou >= 0.25) & (iou < 0.5)).mean()), 3),
      '| 0-0.25', round(float(((iou > 0) & (iou < 0.25)).mean()), 3), '| no box within 1 m', round(float((iou == 0).mean()), 3))
M = [(b, pb) for b, (i, pb, s), *_ in rows if pb is not None]
G = np.array([b for b, _ in M]); Q = np.array([pb for _, pb in M])
u = G[:, :2] / np.linalg.norm(G[:, :2], axis=1, keepdims=True)
dxy = Q[:, :2] - G[:, :2]
rad = (dxy * u).sum(1); tan = dxy[:, 0] * -u[:, 1] + dxy[:, 1] * u[:, 0]
q = lambda a: np.round(np.percentile(a, [10, 50, 90]), 3).tolist()
print('matched', len(M), '| radial offset p10/50/90', q(rad), '| tangential', q(tan), '| |xy| ', q(np.hypot(*dxy.T)))
print('dz', q(Q[:, 2] - G[:, 2]), '| bottom', q((Q[:, 2] - Q[:, 5] / 2) - (G[:, 2] - G[:, 5] / 2)), '| top', q((Q[:, 2] + Q[:, 5] / 2) - (G[:, 2] + G[:, 5] / 2)))
print('label l,w,h p10/50/90', q(G[:, 3]), q(G[:, 4]), q(G[:, 5]))
print('pred  l,w,h p10/50/90', q(Q[:, 3]), q(Q[:, 4]), q(Q[:, 5]))
print('label types', {t: sum(1 for r in rows if r[4] == t) for t in sorted({r[4] for r in rows})})
for lo, hi in [(0, 15), (15, 25), (25, 35), (35, 50), (50, 200)]:
    m = np.array([lo <= r[2] < hi for r in rows])
    if m.sum():
        print(f'range {lo:3d}-{hi:3d} m  n {int(m.sum()):5d}  hit {float((iou[m] >= 0.5).mean()):.2f}  loose {float(((iou[m] >= 0.25) & (iou[m] < 0.5)).mean()):.2f}  far/none {float((iou[m] < 0.25).mean()):.2f}')
pts = np.array([r[3] for r in rows])
for lo, hi in [(5, 10), (10, 20), (20, 50), (50, 150), (150, 100000)]:
    m = (pts >= lo) & (pts < hi)
    if m.sum():
        print(f'points {lo:4d}-{hi:6d}  n {int(m.sum()):5d}  hit {float((iou[m] >= 0.5).mean()):.2f}  loose {float(((iou[m] >= 0.25) & (iou[m] < 0.5)).mean()):.2f}  far/none {float((iou[m] < 0.25).mean()):.2f}')

