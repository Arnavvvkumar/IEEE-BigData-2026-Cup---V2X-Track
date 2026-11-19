"""Label-free post-processing ideas scored on dev against sensor-only labels and cooperative labels with >=5 sensor points
(the closest proxy to the organizers' test labels). Halves = selection / confirmation.
 * gate_ped(k): pedestrian score *= min(1, n / k), n = non-ground points inside the predicted box grown 0.1 m.
 * hedge(f): every vehicle box is also emitted as a truck at score*f, every truck as a vehicle at score*f (class-confusion hedge).
usage: post_tests.py <coop gt json> <calibrated dev predictions.json>"""
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
C = ['vehicle', 'pedestrian', 'truck']
coop = json.loads(Path(sys.argv[1]).read_text())
old = json.loads((KIT / 'ground_truth.json').read_text())
split = json.loads((W / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json').read_text())
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
pred0 = json.loads(Path(sys.argv[2]).read_text())
labels = {'sensor-only': old, 'coop>=5': {f: [g for g in v if g['points'] >= 5] for f, v in coop.items()}}
PLANE = np.array([-0.006, -0.1129, -2.8702])
PTS = {}
for f in ids:
    p = np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3).astype(np.float64)
    p = p[np.abs(p).sum(1) > 0]
    PTS[f] = p[p[:, 2] - (p[:, :2] @ PLANE[:2] + PLANE[2]) > 0.12]


def npts(f, b, grow=0.1):
    p = PTS[f]
    q = p[np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) < max(b[3], b[4]) + 0.5] - np.array(b[:3])
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    x = q[:, 0] * cs - q[:, 1] * sn
    y = q[:, 0] * sn + q[:, 1] * cs
    return int(((np.abs(x) <= b[3] / 2 + grow) & (np.abs(y) <= b[4] / 2 + grow) & (np.abs(q[:, 2]) <= b[5] / 2 + grow)).sum())


NP = {f: [npts(f, d['box']) if d['class'] == 'pedestrian' else -1 for d in pred0[f]] for f in ids}


def gate(k):
    return {f: [{**d, 'score': d['score'] * min(1.0, NP[f][i] / k) if d['class'] == 'pedestrian' else d['score']} for i, d in enumerate(pred0[f])] for f in ids}


def hedge(fr):
    out = {}
    for f in ids:
        dets = list(pred0[f])
        for d in pred0[f]:
            if d['class'] in ('vehicle', 'truck'):
                dets.append({**d, 'class': 'truck' if d['class'] == 'vehicle' else 'vehicle', 'score': d['score'] * fr})
        out[f] = dets
    return out


def score(pred, g):
    row = {}
    for name, sel in (('all', ids), ('sel', split['selection_ids']), ('conf', split['confirmation_ids'])):
        m = ev.evaluate_3d_detection({f: pred[f] for f in sel}, {f: [{'class': x['class'], 'box': x['box']} for x in g[f]] for f in sel}, C, iou_thresholds=(0.5,), iou_type='3d')
        row[name] = {c: round(100 * m['per_class'][c]['AP_0.5_R40'], 1) for c in C}
    return row


V = {'base': pred0, **{f'gate_ped_{k}': gate(k) for k in (3, 5, 8, 12)}, **{f'hedge_{f}': hedge(f) for f in (0.1, 0.3)}}
for name, p in V.items():
    for ln, g in labels.items():
        r = score(p, g)
        cls = ('pedestrian',) if name.startswith('gate') else ('vehicle', 'truck') if name.startswith('hedge') else C
        print(f'{name:12s} {ln:11s}', ' | '.join(f"{c[:4]} all {r['all'][c]:5.1f} sel {r['sel'][c]:5.1f} conf {r['conf'][c]:5.1f}" for c in cls), flush=True)
