"""Label-free look at the 50 test frames: which truck / vehicle boxes sit at the same place in most frames (parked objects)?
With 124 truck labels in 50 frames, a few fixed objects handled badly in every frame would pin truck AP.
Clusters box centres on a 2 m grid across frames; prints clusters present in >= 60% of frames.
Also prints the same table for a validation set, where each cluster can be compared with the labels at that place.
usage: persistent_test.py <dev set dir> <dev norule pred name>"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
F = W / 'experiments/claude_carla_overnight/final'
sys.path.insert(0, str(F))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
test = json.loads((F / 'candidate_v6/predictions.json').read_text())
dev = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))
G = json.loads((d / 'gt_coop.json').read_text())


def clusters(pred, c, smin):
    cells = defaultdict(list)
    for f in pred:
        best = {}
        for x in pred[f]:
            if x['class'] == c and x['score'] >= smin:
                k = (round(x['box'][0] / 2), round(x['box'][1] / 2))
                if k not in best or x['score'] > best[k]['score']:
                    best[k] = x
        for k, x in best.items():
            cells[k].append(x)
    out = []
    for k, v in cells.items():
        if len(v) >= 0.6 * len(pred):
            b = np.median(np.array([x['box'] for x in v]), 0)
            out.append((len(v) / len(pred), float(np.median([x['score'] for x in v])), b))
    return sorted(out, key=lambda r: -r[1])


def gt_at(b, c):
    hits, ious = 0, []
    for f, v in G.items():
        near = [g for g in v if g['class'] == c and g['points'] >= 5 and np.hypot(g['box'][0] - b[0], g['box'][1] - b[1]) < 2.5]
        if near:
            hits += 1
            ious.append(max(ev._iou_3d(np.array(g['box'], float), b) for g in near))
    typ = sorted({g.get('type', '?') for v in G.values() for g in v if g['class'] == c and np.hypot(g['box'][0] - b[0], g['box'][1] - b[1]) < 2.5})
    return hits / len(G), (float(np.median(ious)) if ious else 0.0), typ


for c, smin in (('truck', 0.05), ('vehicle', 0.15)):
    print(f'=== {c}: fixed-place boxes (score >= {smin}) in >= 60% of frames')
    print('  TEST   frames  score   x      y     range   L    W    H')
    for fr, s, b in clusters(test, c, smin):
        print(f'        {100 * fr:4.0f}%  {s:5.2f}  {b[0]:6.1f} {b[1]:6.1f}  {np.hypot(b[0], b[1]):5.1f}  {b[3]:4.1f} {b[4]:4.1f} {b[5]:4.1f}')
    print('  DEV    frames  score   x      y     range   L    W    H   | label there in  median IoU  label types')
    for fr, s, b in clusters(dev, c, smin):
        h, i, t = gt_at(b, c)
        print(f'        {100 * fr:4.0f}%  {s:5.2f}  {b[0]:6.1f} {b[1]:6.1f}  {np.hypot(b[0], b[1]):5.1f}  {b[3]:4.1f} {b[4]:4.1f} {b[5]:4.1f}  | {100 * h:4.0f}% of frames   {i:.2f}   {t}')
print('=== dev truck labels by type:', {t: sum(g.get('type') == t for v in G.values() for g in v if g['class'] == 'truck' and g['points'] >= 5) for t in sorted({g.get('type', '?') for v in G.values() for g in v if g['class'] == 'truck'})})
print('test truck boxes per frame at score >= 0.05 / 0.15 / 0.3:', [round(sum(x['class'] == 'truck' and x['score'] >= t for f in test for x in test[f]) / len(test), 2) for t in (0.05, 0.15, 0.3)])
