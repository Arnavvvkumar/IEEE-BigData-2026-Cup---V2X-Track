"""Test-time use of the static site map (points only, no labels in the method; labels only score it).
Cross-fitted: the static map applied to a frame is built from the OTHER half of the dev frames (selection <-> confirmation),
so no frame is filtered with a map that contains it.
Variants: drop / down-weight detections sitting on static structure, and snap pedestrian boxes to their own points + ground.
usage: static_filter.py <predictions.json (sensor frame)> <out.json>"""
import os
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/erroranalysis'))
from groundplane import fit_ground, level_transform  # noqa: E402

KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
gt = json.loads((KIT / 'ground_truth.json').read_text())
split = json.loads((W / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json').read_text())
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
C = ['vehicle', 'pedestrian', 'truck']
V, GZ, GCUT = 0.25, -3.0, 0.25
pred0 = json.loads(Path(sys.argv[1]).read_text())

clouds = {f: np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
planes = np.array([fit_ground(clouds[f], np.random.default_rng(i)) for i, f in enumerate(ids)])
R, t = level_transform(np.median(planes, 0), GZ)
lev = {}
for f in ids:
    p = clouds[f] @ R.T + t
    lev[f] = p[p[:, 2] > GZ + GCUT]  # non-ground points, levelled frame


def vkey(p):
    k = np.floor(p / V).astype(np.int64) + 2000
    return k[:, 0] * 4096 * 4096 + k[:, 1] * 4096 + k[:, 2]


def static_set(frames, frac):
    allk = np.concatenate([np.unique(vkey(lev[f])) for f in frames])
    u, c = np.unique(allk, return_counts=True)
    return u[c >= frac * len(frames)]


def score(pred):
    row = {}
    for name, sel in (('all', ids), ('sel', split['selection_ids']), ('conf', split['confirmation_ids'])):
        m = ev.evaluate_3d_detection({f: pred[f] for f in sel}, {f: gt[f] for f in sel}, C, iou_thresholds=(0.5,), iou_type='3d')
        row[name] = round(100 * m['mAP_0.5_R40'], 2)
        if name == 'all':
            row.update({c: round(100 * m['per_class'][c]['AP_0.5_R40'], 2) for c in C})
    return row


def to_level(box):
    b = np.array(box, float)
    c = R @ b[:3] + t
    h = R @ np.array([np.cos(b[6]), np.sin(b[6]), 0.0])
    return c, b[3:6], np.arctan2(h[1], h[0])


def in_box(p, c, d, yaw, grow=0.0):
    q = p - c
    cs, sn = np.cos(-yaw), np.sin(-yaw)
    x = q[:, 0] * cs - q[:, 1] * sn
    y = q[:, 0] * sn + q[:, 1] * cs
    return (np.abs(x) <= d[0] / 2 + grow) & (np.abs(y) <= d[1] / 2 + grow) & (np.abs(q[:, 2]) <= d[2] / 2 + grow)


def features(frac):
    """per detection: n non-ground points in box, static fraction, snapped box (pedestrians)."""
    S = {'sel': static_set(split['confirmation_ids'], frac), 'conf': static_set(split['selection_ids'], frac)}
    feats = {}
    for half in ('sel', 'conf'):
        for f in split['selection_ids' if half == 'sel' else 'confirmation_ids']:
            p = lev[f]
            is_static = np.isin(vkey(p), S[half])
            rows = []
            for d in pred0[f]:
                c, dims, yaw = to_level(d['box'])
                m = in_box(p, c, dims, yaw)
                n = int(m.sum())
                sf = float(is_static[m].mean()) if n else 0.0
                snap = None
                if d['class'] == 'pedestrian':
                    near = in_box(p, c, dims, yaw, grow=0.25) & ~is_static
                    if near.sum() >= 4:
                        cc = c.copy()
                        cc[:2] = np.median(p[near, :2], 0)
                        cc[2] = GZ + dims[2] / 2
                        back = (cc - t) @ R
                        snap = back.tolist() + list(d['box'][3:])
                rows.append((n, sf, snap))
            feats[f] = rows
    return feats


def variant(feats, drop=None, scale=None, classes=('pedestrian',), snap=False, empty=None):
    out = {}
    for f in ids:
        dets = []
        for d, (n, sf, sb) in zip(pred0[f], feats[f]):
            d = dict(d)
            if d['class'] in classes:
                if drop is not None and n >= 3 and sf >= drop:
                    continue
                if scale is not None:
                    d['score'] = d['score'] * (1 - scale * sf)
                if empty is not None and n < 2:
                    d['score'] = d['score'] * empty
            if snap and sb is not None:
                d['box'] = sb
            dets.append(d)
        out[f] = dets
    return out


res = {'base': score(pred0)}
print('base', json.dumps(res['base']), flush=True)
best = ('base', res['base']['sel'], pred0)
for frac in (0.3, 0.5):
    F = features(frac)
    grid = {
        f'ped_drop0.6_f{frac}': dict(drop=0.6), f'ped_drop0.8_f{frac}': dict(drop=0.8),
        f'ped_scale0.8_f{frac}': dict(scale=0.8), f'ped_empty0.3_f{frac}': dict(empty=0.3),
        f'all_scale0.8_f{frac}': dict(scale=0.8, classes=tuple(C)), f'all_drop0.8_f{frac}': dict(drop=0.8, classes=tuple(C)),
        f'ped_snap_f{frac}': dict(snap=True), f'ped_snap_scale0.8_empty0.3_f{frac}': dict(snap=True, scale=0.8, empty=0.3),
    }
    for name, kw in grid.items():
        pv = variant(F, **kw)
        res[name] = score(pv)
        print(name, json.dumps(res[name]), flush=True)
        if res[name]['sel'] > best[1]:
            best = (name, res[name]['sel'], pv)
res['best_by_selection'] = best[0]
out = Path(sys.argv[2])
out.write_text(json.dumps(res, indent=1))
out.with_name(out.stem + '_predictions.json').write_text(json.dumps(best[2]))
print('best by selection', best[0])
