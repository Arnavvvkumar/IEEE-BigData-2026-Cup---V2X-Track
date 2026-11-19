"""Class-confusion hedge on the 150-frame validation set: each vehicle box is also emitted as a truck at score*f (and the
reverse at score*g). Reports vehicle / truck AP, and truck AP on 'genuine' trucks only (TrashCan boxes and the predictions
that hit them removed), since the test set has about twice as many genuine trucks per frame as our validation frames.
usage: hedge150.py <set dir>"""
import os
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d = Path(sys.argv[1])
gt = {f: [g for g in v if g['points'] >= 5] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
pred = json.loads((d / 'pred_cw_d_calibrated.json').read_text())
ids = sorted(gt)
halves = {'all': ids, 'A': ids[0::2], 'B': ids[1::2]}


def ap(p, g, cls):
    out = []
    for h in halves.values():
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: [{'class': x['class'], 'box': x['box']} for x in g[f]] for f in h}, [cls], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class'][cls]['AP_0.5_R40'], 1))
    return out


def hedge(fv, ft, size_gate=None):
    out = {}
    for f in ids:
        dets = list(pred[f])
        for x in pred[f]:
            if x['class'] == 'vehicle' and fv and (size_gate is None or x['box'][3] >= size_gate):
                dets.append({**x, 'class': 'truck', 'score': x['score'] * fv})
            if x['class'] == 'truck' and ft:
                dets.append({**x, 'class': 'vehicle', 'score': x['score'] * ft})
        out[f] = dets
    return out


def genuine(p):
    """drop TrashCan labels and any truck prediction overlapping one (IoU >= 0.3), to look at genuine trucks alone."""
    g2, p2 = {}, {}
    for f in ids:
        cans = [np.array(g['box']) for g in gt[f] if g['type'] == 'TrashCan']
        g2[f] = [g for g in gt[f] if g['type'] != 'TrashCan']
        p2[f] = [x for x in p[f] if not (x['class'] == 'truck' and any(ev._iou_3d(np.array(x['box']), c) >= 0.3 for c in cans))]
    return p2, g2


print('variant                 vehicle all/A/B      truck all/A/B        genuine-truck all/A/B')
for name, kw in (('base', dict(fv=0, ft=0)), ('veh->truck 0.3', dict(fv=0.3, ft=0)), ('veh->truck 0.5', dict(fv=0.5, ft=0)), ('veh->truck 0.8', dict(fv=0.8, ft=0)),
                 ('veh>=5.2m->truck 0.8', dict(fv=0.8, ft=0, size_gate=5.2)), ('veh>=4.9m->truck 0.8', dict(fv=0.8, ft=0, size_gate=4.9)),
                 ('both 0.5 / 0.2', dict(fv=0.5, ft=0.2)), ('truck->veh 0.3', dict(fv=0, ft=0.3))):
    p = hedge(**kw)
    pg, gg = genuine(p)
    print(f'{name:22s}', ap(p, gt, 'vehicle'), ap(p, gt, 'truck'), ap(pg, gg, 'truck'), flush=True)
