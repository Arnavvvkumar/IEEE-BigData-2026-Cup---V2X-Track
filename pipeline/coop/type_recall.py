"""Where do the dev truck / pedestrian APs come from? Hit rate of our predictions by raw V2X-Real object type (centre within
1.5 m / 0.6 m of a same-class prediction with score >= 0.05), plus sizes. usage: type_recall.py <coop gt> <predictions>"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

gt = json.loads(Path(sys.argv[1]).read_text())
pred = json.loads(Path(sys.argv[2]).read_text())
by = defaultdict(list)
for f, v in gt.items():
    for g in v:
        if g['points'] < 5:
            continue
        rad = {'vehicle': 1.5, 'truck': 1.5, 'pedestrian': 0.6}[g['class']]
        ps = [d for d in pred[f] if d['class'] == g['class'] and d['score'] >= 0.05 and np.hypot(d['box'][0] - g['box'][0], d['box'][1] - g['box'][1]) < rad]
        best = max(ps, key=lambda d: d['score']) if ps else None
        by[(g['class'], g['type'])].append((best is not None, best['score'] if best else 0.0, g['box'][3:6], g['points'], f))
print('class/type            n  hit-rate  median best score  median GT l,w,h        median pts   frames')
for (c, t), rows in sorted(by.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
    hit = np.mean([r[0] for r in rows])
    sc = np.median([r[1] for r in rows])
    sz = np.median([r[2] for r in rows], 0).round(2).tolist()
    print(f'{c[:4]}/{t:14s} {len(rows):4d}   {hit:5.2f}     {sc:5.2f}              {sz}   {int(np.median([r[3] for r in rows])):5d}   {len({r[4] for r in rows})}')
