"""What are the false positives? For each predicted box of a class (by score band): true positive, duplicate of an already-matched
label, loose box on a same-class label, box on a label of another class, box on a label with < 5 points (not scored), or on nothing.
usage: fp_anatomy.py <set dir> <pred json name>"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d = Path(sys.argv[1])
pred = json.loads((d / sys.argv[2]).read_text())
gt = json.loads((d / 'gt_coop.json').read_text())
bands = [(0.5, 2), (0.3, 0.5), (0.1, 0.3), (0.03, 0.1)]
out = {c: {b: Counter() for b in bands} for c in ('vehicle', 'truck', 'pedestrian')}
for f in gt:
    G = [(np.array(g['box'], float), g['class'], g['points'] >= 5, g.get('type', '')) for g in gt[f]]
    for c in out:
        used = set()
        for x in sorted((x for x in pred[f] if x['class'] == c and x['score'] >= 0.03), key=lambda x: -x['score']):
            b = np.array(x['box'], float)
            band = next(bd for bd in bands if bd[0] <= x['score'] < bd[1])
            near = [(ev._iou_3d(b, gb), j) for j, (gb, *_r) in enumerate(G) if np.hypot(b[0] - gb[0], b[1] - gb[1]) < 8]
            same = sorted(((i, j) for i, j in near if G[j][1] == c and G[j][2]), reverse=True)
            if same and same[0][0] >= 0.5 and same[0][1] not in used:
                used.add(same[0][1]); kind = 'TP'
            elif same and same[0][0] >= 0.5:
                kind = 'duplicate'
            elif same and same[0][0] >= 0.1:
                kind = 'loose on same class'
            else:
                oth = max(((i, j) for i, j in near if G[j][1] != c and G[j][2]), default=(0, -1))
                few = max(((i, j) for i, j in near if not G[j][2]), default=(0, -1))
                if oth[0] >= 0.1:
                    kind = f'on {G[oth[1]][1]}/{G[oth[1]][3]}'
                elif few[0] >= 0.1:
                    kind = 'on label with <5 pts'
                else:
                    kind = 'on nothing'
            out[c][band][kind] += 1
for c in out:
    print('==', c)
    for bd in bands:
        n = sum(out[c][bd].values())
        if n:
            print(f'  score {bd[0]}-{min(bd[1], 1)}: n {n:6d} | ' + ', '.join(f'{k} {v / n:.2f}' for k, v in out[c][bd].most_common(7)))
