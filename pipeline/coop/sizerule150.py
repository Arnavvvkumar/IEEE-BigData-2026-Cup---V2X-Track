"""Size rule for the truck class (thresholds from real box-size statistics, no fitted model): a VEHICLE prediction whose box is
truck-like (at least K of: length >= L, width >= Wd, height >= H) is also emitted as a truck at score*a, and its vehicle copy
is kept at score*b. Reports vehicle / truck AP and truck AP on genuine trucks only (fixed 'TrashCan' object removed).
usage: sizerule150.py <set dir>"""
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
gt = {f: [g for g in v if g['points'] >= 5] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
pred = json.loads((d / 'pred_cw_d_calibrated.json').read_text())
ids = sorted(gt)
halves = {'all': ids, 'A': ids[0::2], 'B': ids[1::2]}
CANS = {f: [np.array(g['box']) for g in gt[f] if g['type'] == 'TrashCan'] for f in ids}
GEN = {f: [g for g in gt[f] if g['type'] != 'TrashCan'] for f in ids}


def ap(p, g, cls):
    out = []
    for h in halves.values():
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: [{'class': x['class'], 'box': x['box']} for x in g[f]] for f in h}, [cls], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class'][cls]['AP_0.5_R40'], 1))
    return out


def rule(K=2, L=5.0, Wd=2.15, H=1.95, a=0.9, b=0.5, base=0.0):
    out = {}
    for f in ids:
        dets = []
        for x in pred[f]:
            if x['class'] == 'vehicle':
                l, w, h = x['box'][3:6]
                like = (l >= L) + (w >= Wd) + (h >= H) >= K
                if like:
                    dets.append({**x, 'class': 'truck', 'score': x['score'] * a})
                    dets.append({**x, 'score': x['score'] * b})
                else:
                    dets.append(x)
                    if base:
                        dets.append({**x, 'class': 'truck', 'score': x['score'] * base})
            else:
                dets.append(x)
        out[f] = dets
    return out


def genuine(p):
    return {f: [x for x in p[f] if not (x['class'] == 'truck' and any(ev._iou_3d(np.array(x['box']), c) >= 0.3 for c in CANS[f]))] for f in ids}


print('variant                                   vehicle all/A/B      truck all/A/B        genuine-truck all/A/B')
grid = {'base': None,
        'K2 a0.9 b0.5': dict(), 'K2 a0.9 b0.8': dict(b=0.8), 'K2 a0.9 b1.0': dict(b=1.0), 'K2 a0.6 b1.0': dict(a=0.6, b=1.0),
        'K2 a0.9 b0.5 +base0.2': dict(base=0.2), 'K1 a0.9 b0.8': dict(K=1, b=0.8), 'K3 a0.9 b0.5': dict(K=3),
        'K2 w2.1 h1.9 a0.9 b0.8': dict(Wd=2.1, H=1.9, b=0.8), 'K2 w2.2 h2.0 a0.9 b0.8': dict(Wd=2.2, H=2.0, b=0.8),
        'K2 a1.0 b0.3': dict(a=1.0, b=0.3)}
for name, kw in grid.items():
    p = pred if kw is None else rule(**kw)
    print(f'{name:40s}', ap(p, gt, 'vehicle'), ap(p, gt, 'truck'), ap(genuine(p), GEN, 'truck'), flush=True)
