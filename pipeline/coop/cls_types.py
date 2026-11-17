"""By-type outcome of every label (hit / wrong class / loose box / missed) for a prediction file, plus false-positive counts.
usage: cls_types.py <set dir> <pred json name>"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d = Path(sys.argv[1])
pred = json.loads((d / sys.argv[2]).read_text())
gt = {f: [g for g in v if g['points'] >= 5] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
rows = defaultdict(list)
fp = defaultdict(lambda: [0, 0])
for f in gt:
    P = [(np.array(x['box']), x['score'], x['class']) for x in pred[f] if x['score'] >= 0.05]
    for g in gt[f]:
        if g['class'] == 'pedestrian':
            continue
        b = np.array(g['box'])
        near = [(pb, s, c) for pb, s, c in P if np.hypot(pb[0] - b[0], pb[1] - b[1]) < max(b[3], 2.0)]
        same = max([(ev._iou_3d(pb, b), s, pb) for pb, s, c in near if c == g['class']], default=(0.0, 0.0, None), key=lambda t: t[0])
        other = max([(ev._iou_3d(pb, b), s, c) for pb, s, c in near if c != g['class'] and c != 'pedestrian'], default=(0.0, 0.0, ''), key=lambda t: t[0])
        kind = 'hit' if same[0] >= 0.5 else 'wrong class' if other[0] >= 0.5 else 'loose box' if max(same[0], other[0]) >= 0.25 else 'missed'
        rows[(g['class'], g['type'])].append((kind, np.hypot(b[0], b[1]), g['points'], b[3], same[2][3] if same[2] is not None else np.nan))
    G = [(np.array(g['box']), g['class']) for g in gt[f]]
    for pb, s, c in P:
        if c == 'pedestrian' or s < 0.3:
            continue
        best = max([ev._iou_3d(pb, gb) for gb, gc in G if np.hypot(pb[0] - gb[0], pb[1] - gb[1]) < 6], default=0.0)
        fp[c][0] += best < 0.1
        fp[c][1] += 1
print(f"{'class/type':24s}    n   hit  wrong  loose  missed | label L med | pred L med (loose) | missed: range, pts")
for (c, t), r in sorted(rows.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
    k = [x[0] for x in r]
    miss = [x for x in r if x[0] == 'missed']
    loose = [x[4] for x in r if x[0] == 'loose box' and np.isfinite(x[4])]
    print(f"{c[:4] + '/' + t:24s} {len(r):4d}  {k.count('hit') / len(r):4.2f}  {k.count('wrong class') / len(r):4.2f}   {k.count('loose box') / len(r):4.2f}   {k.count('missed') / len(r):4.2f}  |   {np.median([x[3] for x in r]):5.2f}    |   {np.median(loose) if loose else 0:5.2f}   |"
          + (f" {np.median([x[1] for x in miss]):5.1f} m, {int(np.median([x[2] for x in miss]))} pts" if miss else ''))
print('confident (score>=0.3) boxes on nothing (IoU<0.1 with any label):', {c: f'{v[0]}/{v[1]}' for c, v in fp.items()})
