"""Where do pedestrian boxes lose IoU on the real dev frames? (diagnosis; labels used only to score / to build oracles)
Oracle variants replace ONE component of each matched pedestrian prediction with the label's value (never submitted);
label-free variants test simple conventions (yaw, radial / vertical shift, size scale).
usage: ped_decomp.py <predictions.json> <out.json>"""
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


def score(pred):
    row = {}
    for name, sel in (('all', ids), ('sel', split['selection_ids']), ('conf', split['confirmation_ids'])):
        m = ev.evaluate_3d_detection({f: pred[f] for f in sel}, {f: GT[f] for f in sel}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        row[name] = round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 2)
    return row


def match(f):
    """index of the nearest label (BEV centre within 0.6 m) for each prediction, else -1."""
    g = np.array([x['box'] for x in GT[f]]).reshape(-1, 7)
    out = []
    for d in PED[f]:
        if not len(g):
            out.append(-1)
            continue
        dist = np.hypot(g[:, 0] - d['box'][0], g[:, 1] - d['box'][1])
        j = int(dist.argmin())
        out.append(j if dist[j] < 0.6 else -1)
    return g, out


def variant(fn):
    out = {}
    for f in ids:
        g, m = match(f)
        dets = []
        for d, j in zip(PED[f], m):
            b = np.array(d['box'], float)
            b = fn(b, g[j] if j >= 0 else None)
            dets.append({'box': b.tolist(), 'score': d['score'], 'class': 'pedestrian'})
        out[f] = dets
    return out


def oracle(idx):
    def fn(b, g):
        if g is not None:
            b = b.copy()
            b[idx] = g[idx]
        return b
    return fn


def radial(delta):
    def fn(b, g):
        b = b.copy()
        r = np.hypot(b[0], b[1])
        b[:2] *= (r + delta) / max(r, 1e-6)
        return b
    return fn


def edit(**kw):
    def fn(b, g):
        b = b.copy()
        if 'dz' in kw:
            b[2] += kw['dz']
        if 'yaw' in kw:
            b[6] = np.arctan2(b[1], b[0]) + kw['yaw'] if kw.get('rel') else kw['yaw']
        if 'lw' in kw:
            b[3:5] *= kw['lw']
        if 'h' in kw:
            b[5] *= kw['h']
        if kw.get('square'):
            b[3:5] = (b[3] + b[4]) / 2
        if kw.get('ground'):
            b[2] = PLANE[0] * b[0] + PLANE[1] * b[1] + PLANE[2] + b[5] / 2 + kw['ground_off']
        return b
    return fn


V = {'base': lambda b, g: b,
     'oracle_xy': oracle([0, 1]), 'oracle_z': oracle([2]), 'oracle_dims': oracle([3, 4, 5]), 'oracle_yaw': oracle([6]),
     'oracle_xyz': oracle([0, 1, 2]), 'oracle_z_dims_yaw': oracle([2, 3, 4, 5, 6]), 'oracle_h_z': oracle([2, 5]), 'oracle_lw_yaw': oracle([3, 4, 6]),
     'yaw_0': edit(yaw=0.0), 'yaw_45': edit(yaw=np.pi / 4), 'yaw_90': edit(yaw=np.pi / 2), 'yaw_radial': edit(yaw=0.0, rel=True),
     'yaw_radial45': edit(yaw=np.pi / 4, rel=True), 'square': edit(square=True),
     **{f'radial_{d:+.2f}': radial(d) for d in (-0.15, -0.1, -0.05, 0.05, 0.1, 0.15)},
     **{f'dz_{d:+.2f}': edit(dz=d) for d in (-0.15, -0.1, -0.05, 0.05, 0.1, 0.15)},
     **{f'lw_x{s}': edit(lw=s) for s in (0.85, 1.15, 1.3)}, **{f'h_x{s}': edit(h=s) for s in (0.9, 1.1)},
     **{f'ground_{o:+.2f}': edit(ground=True, ground_off=o) for o in (-0.1, 0.0, 0.1)}}
res = {}
# label statistics for context (pedestrian boxes): sizes and yaw convention
G = np.concatenate([np.array([x['box'] for x in GT[f]]).reshape(-1, 7) for f in ids])
bear = np.arctan2(G[:, 1], G[:, 0])
rel = (G[:, 6] - bear + np.pi / 4) % (np.pi / 2) - np.pi / 4  # yaw relative to the sensor ray, folded to +-45 deg
absy = (G[:, 6] + np.pi / 4) % (np.pi / 2) - np.pi / 4
res['label_stats'] = {'n': len(G), 'lwh_median': np.median(G[:, 3:6], 0).round(3).tolist(), 'lwh_std': G[:, 3:6].std(0).round(3).tolist(),
                      'yaw_folded_abs_deg_hist': np.histogram(np.degrees(absy), bins=6, range=(-45, 45))[0].tolist(),
                      'yaw_folded_radial_deg_hist': np.histogram(np.degrees(rel), bins=6, range=(-45, 45))[0].tolist(),
                      'z_minus_plane_minus_halfh_median': float(np.median(G[:, 2] - (G[:, :2] @ PLANE[:2] + PLANE[2]) - G[:, 5] / 2).round(3))}
print('label_stats', json.dumps(res['label_stats']), flush=True)
P = np.concatenate([np.array([d['box'] for d in PED[f] if d['score'] > 0.15]).reshape(-1, 7) for f in ids])
print('pred lwh median (score>0.15)', np.median(P[:, 3:6], 0).round(3).tolist(), flush=True)
for name, fn in V.items():
    res[name] = score(variant(fn))
    print(name, json.dumps(res[name]), flush=True)
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
