"""Per-class box calibration on the real dev frames: size offsets (box-size statistics), vertical and radial shifts.
Each class is scored alone (per-class AP is independent). Parameters are judged on the selection half and checked on the
confirmation half. usage: class_adjust.py <predictions.json> <out.json> <class>"""
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
CLS = sys.argv[3]
P = {f: [d for d in pred0[f] if d['class'] == CLS] for f in ids}
GT = {f: [g for g in gt[f] if g['class'] == CLS] for f in ids}


def adjust(b, dl=0.0, dw=0.0, dh=0.0, dz=0.0, rad=0.0, sl=1.0):
    b = np.array(b, float)
    b[3] = b[3] * sl + dl
    b[4] += dw
    b[5] += dh
    b[2] += dz
    if rad:
        r = np.hypot(b[0], b[1])
        b[:2] *= (r + rad) / max(r, 1e-6)
    return b


def score(pred):
    row = {}
    for name, sel in (('all', ids), ('sel', split['selection_ids']), ('conf', split['confirmation_ids'])):
        m = ev.evaluate_3d_detection({f: pred[f] for f in sel}, {f: GT[f] for f in sel}, [CLS], iou_thresholds=(0.5,), iou_type='3d')
        row[name] = round(100 * m['per_class'][CLS]['AP_0.5_R40'], 2)
    return row


if CLS == 'vehicle':
    grid = {'base': {}, 'size': dict(dl=0.16, dw=0.08, dh=0.09), 'size_z': dict(dl=0.16, dw=0.08, dh=0.09, dz=0.11),
            'z0.06': dict(dz=0.06), 'z0.11': dict(dz=0.11), 'size_z_rad': dict(dl=0.16, dw=0.08, dh=0.09, dz=0.11, rad=-0.08),
            'rad-0.08': dict(rad=-0.08), 'size_half_z': dict(dl=0.08, dw=0.04, dh=0.09, dz=0.08), 'h_z': dict(dh=0.09, dz=0.11),
            'size_big_z': dict(dl=0.3, dw=0.12, dh=0.09, dz=0.11)}
elif CLS == 'truck':
    grid = {'base': {}, 'dl0.3': dict(dl=0.3), 'dl0.5': dict(dl=0.5), 'sl1.07': dict(sl=1.07), 'size': dict(dl=0.5, dw=0.1, dh=0.1),
            'size_z': dict(dl=0.5, dw=0.1, dh=0.1, dz=0.03), 'rad-0.2': dict(rad=-0.2), 'rad-0.35': dict(rad=-0.35),
            'dl0.5_rad-0.2': dict(dl=0.5, rad=-0.2)}
else:
    grid = {'base': {}, 'rad-0.045': dict(rad=-0.045), 'rad-0.045_h': dict(rad=-0.045, dh=-0.047, dz=-0.018),
            'rad-0.045_lw': dict(rad=-0.045, dl=-0.028, dw=-0.028), 'rad-0.045_z-0.02': dict(rad=-0.045, dz=-0.02)}
res = {}
for name, kw in grid.items():
    pv = {f: [{'box': adjust(d['box'], **kw).tolist(), 'score': d['score'], 'class': CLS} for d in P[f]] for f in ids}
    res[name] = {**score(pv), 'params': kw}
    print(CLS, name, json.dumps({k: v for k, v in res[name].items() if k != 'params'}), flush=True)
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
