"""Pedestrian box-centre refinement on the real dev frames (post-processing; parameters chosen on the selection half,
checked on the confirmation half). Label-free at run time: uses only the predicted box and the lidar points.
 * radial shift s: move the centre s metres along the sensor ray (negative = towards the sensor).
 * blend w: centre = w * shifted prediction + (1 - w) * (centroid of non-ground points in the box + c along the ray).
usage: ped_refine.py <predictions.json> <out.json>"""
import os
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
gt = json.loads((KIT / 'ground_truth.json').read_text())
split = json.loads((W / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json').read_text())
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
pred0 = json.loads(Path(sys.argv[1]).read_text())
PED = {f: [d for d in pred0[f] if d['class'] == 'pedestrian'] for f in ids}
GT = {f: [g for g in gt[f] if g['class'] == 'pedestrian'] for f in ids}
PLANE = np.array([-0.006, -0.1129, -2.8702])
PTS = {}
for f in ids:
    p = np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3).astype(np.float64)
    p = p[np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)]
    PTS[f] = p[p[:, 2] - (p[:, :2] @ PLANE[:2] + PLANE[2]) > 0.12]


def refine_box(b, p, s=0.0, w=1.0, c=0.065, grow=0.15, min_pts=5, zmode=None):
    b = np.array(b, float)
    r = np.hypot(b[0], b[1])
    u = b[:2] / max(r, 1e-6)
    b[:2] += s * u
    if w < 1.0:
        q = p[np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) < 1.0] - b[:3]
        cs, sn = np.cos(-b[6]), np.sin(-b[6])
        x = q[:, 0] * cs - q[:, 1] * sn
        y = q[:, 0] * sn + q[:, 1] * cs
        m = (np.abs(x) <= b[3] / 2 + grow) & (np.abs(y) <= b[4] / 2 + grow) & (np.abs(q[:, 2]) <= b[5] / 2 + grow)
        if m.sum() >= min_pts:
            cen = q[m, :2].mean(0) + b[:2] + c * u
            b[:2] = w * b[:2] + (1 - w) * cen
    return b


def score(pred):
    row = {}
    for name, sel in (('all', ids), ('sel', split['selection_ids']), ('conf', split['confirmation_ids'])):
        m = ev.evaluate_3d_detection({f: pred[f] for f in sel}, {f: GT[f] for f in sel}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        row[name] = round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 2)
    return row


def variant(**kw):
    return {f: [{'box': refine_box(d['box'], PTS[f], **kw).tolist(), 'score': d['score'], 'class': 'pedestrian'} for d in PED[f]] for f in ids}


if __name__ == '__main__':
    grid = {'base': {}}
    for s in (-0.03, -0.045, -0.06, -0.075):
        grid[f'shift{s}'] = dict(s=s)
    for w in (0.0, 0.3, 0.5, 0.7):
        for c in (0.05, 0.065, 0.08):
            grid[f'blend_w{w}_c{c}'] = dict(s=-0.045, w=w, c=c)
    grid['blend_w0.5_c0.065_grow0.25'] = dict(s=-0.045, w=0.5, c=0.065, grow=0.25)
    grid['blend_w0.5_c0.065_grow0.05'] = dict(s=-0.045, w=0.5, c=0.065, grow=0.05)
    res = {}
    for name, kw in grid.items():
        res[name] = {**score(variant(**kw)), 'params': kw}
        print(name, json.dumps({k: v for k, v in res[name].items() if k != 'params'}), flush=True)
        Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
    best = max((k for k in res), key=lambda k: res[k]['sel'])
    print('best by selection', best, json.dumps(res[best]))
