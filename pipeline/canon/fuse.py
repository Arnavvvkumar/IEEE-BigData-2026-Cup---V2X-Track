"""Fuse levelled predictions from several ground heights and/or several models (real dev frames, evaluation only).
Each pass is one (model, ground height) prediction set, already mapped back to the sensor frame.
Per frame and class: greedy clustering by BEV centre distance in descending score order; a cluster's box is the
score-weighted mean of its members (yaw from the top member); its score is sum(member scores)/number of passes ('mean')
or the top member's score ('max'). No labels are used by the fusion; labels are only used to score it.
usage: fuse.py <out.json> <allpred.json>[:g1,g2,...] ..."""
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
C = ['vehicle', 'pedestrian', 'truck']
RADIUS = {'vehicle': 1.0, 'pedestrian': 0.4, 'truck': 1.5}


def score(pred):
    row = {}
    for name, sel in (('all', ids), ('sel', split['selection_ids']), ('conf', split['confirmation_ids'])):
        m = ev.evaluate_3d_detection({f: pred[f] for f in sel}, {f: gt[f] for f in sel}, C, iou_thresholds=(0.5,), iou_type='3d')
        row[name] = round(100 * m['mAP_0.5_R40'], 2)
        if name == 'all':
            row.update({c: round(100 * m['per_class'][c]['AP_0.5_R40'], 2) for c in C})
    return row


def fuse(passes, mode, scale=1.0):
    n = len(passes)
    out = {}
    for f in ids:
        dets = []
        for c in C:
            items = sorted((d for p in passes for d in p[f] if d['class'] == c), key=lambda d: -d['score'])
            if not items:
                continue
            boxes = np.array([d['box'] for d in items])
            scores = np.array([d['score'] for d in items])
            used = np.zeros(len(items), bool)
            r = RADIUS[c] * scale
            for i in range(len(items)):
                if used[i]:
                    continue
                member = (~used) & (np.hypot(boxes[:, 0] - boxes[i, 0], boxes[:, 1] - boxes[i, 1]) < r)
                used |= member
                w = scores[member]
                b = (boxes[member, :6] * w[:, None]).sum(0) / w.sum()
                s = w.sum() / n if mode == 'mean' else scores[i]
                dets.append({'box': b.tolist() + [float(boxes[i, 6])], 'score': float(min(s, 1.0)), 'class': c})
        out[f] = dets
    return out


passes, names = [], []
for arg in sys.argv[2:]:
    # <path>[:g1,g2,...[:class1,class2]]  (third field keeps only those classes from this model)
    path, gs, keep = (arg.split(':') + ['', ''])[:3]
    data = json.loads(Path(path).read_text())
    for g in (gs.split(',') if gs else list(data)):
        p = data[g]
        if keep:
            kc = set(keep.split(','))
            p = {f: [d for d in p[f] if d['class'] in kc] for f in p}
        passes.append(p)
        names.append(f'{Path(path).stem}@{g}{"#" + keep if keep else ""}')
res = {'passes': names, 'single': {}, 'fused': {}}
for nm, p in zip(names, passes):
    res['single'][nm] = score(p)
    print('single', nm, json.dumps(res['single'][nm]), flush=True)
best = (None, -1, None)
for mode in ('mean', 'max'):
    for scale in (0.6, 1.0):
        fz = fuse(passes, mode, scale)
        row = score(fz)
        res['fused'][f'{mode}_r{scale}'] = row
        print('fused', mode, scale, json.dumps(row), flush=True)
        if row['sel'] > best[1]:
            best = (f'{mode}_r{scale}', row['sel'], fz)
res['best_by_selection'] = best[0]
out = Path(sys.argv[1])
out.write_text(json.dumps(res, indent=1))
out.with_name(out.stem + '_predictions.json').write_text(json.dumps(best[2]))
