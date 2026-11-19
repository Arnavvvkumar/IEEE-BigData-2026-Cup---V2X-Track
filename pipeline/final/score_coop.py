#!/usr/bin/env python
"""AP@0.5 R40 on a coop validation set, same halves as coop/eval150.py.
usage: score_coop.py <set dir> <predictions.json>
"""
import os
import importlib.util
import json
import sys
from pathlib import Path

W = Path(os.environ['BDC_ROOT'])
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
C = ['vehicle', 'pedestrian', 'truck']
folder, pred_path = Path(sys.argv[1]), Path(sys.argv[2])
gt = {f: [g for g in v if g['points'] >= 5] for f, v in json.loads((folder / 'gt_coop.json').read_text()).items()}
pred = json.loads(pred_path.read_text())
ids = sorted(gt)
halves = [ids, ids[0::2], ids[1::2]]


def ap(p, g, cls):
    out = []
    for h in halves:
        m = ev.evaluate_3d_detection(
            {f: p[f] for f in h},
            {f: [{'class': x['class'], 'box': x['box']} for x in g[f]] for f in h},
            [cls], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class'][cls]['AP_0.5_R40'], 1))
    return out


res = {c: ap(pred, gt, c) for c in C}
res['mAP'] = [round(sum(res[c][i] for c in C) / 3, 2) for i in range(3)]
print(json.dumps(res))
