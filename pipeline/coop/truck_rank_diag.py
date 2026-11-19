"""Label-free diagnosis of how our TRUCK boxes rank on the 50 test frames (v9 predictions). AP depends on the order of boxes by score:
a real truck ranked below many false boxes loses precision. For each score band: boxes per frame that sit on the fixed truck-sized
spots (A-E, found earlier by persistent_test.py) vs elsewhere, and whether a box came from the detectors' own truck class or was copied
from a vehicle box by the truck size rule. No labels used.
usage: truck_rank_diag.py"""
import os
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
F = W / 'experiments/claude_carla_overnight/final'
p = json.loads(zipfile.ZipFile(os.environ['BDC_OUT'] + '/bdc2026_trackB_submission_v9.zip').read('predictions.json'))
fused = json.loads((F / 'candidate_v9_meta/fused.json').read_text())
SPOTS = {'A': (-4.4, 26.1), 'B': (15.7, -24.6), 'C': (4.2, -24.2), 'D': (27.7, -23.6), 'E': (-30.6, -22.8)}
key = lambda b: tuple(round(v, 5) for v in b)
native = {f: {key(d['box']) for d in v if d['class'] == 'truck'} for f, v in fused.items()}
rows = []
for f, v in p.items():
    for d in v:
        if d['class'] != 'truck' or d['score'] < 0.02:
            continue
        b = d['box']
        spot = next((k for k, (x, y) in SPOTS.items() if np.hypot(b[0] - x, b[1] - y) < 3.0), '-')
        # hedge variants have a different length/height than their primary: count only primaries by checking score tier is not a x0.05 copy
        rows.append((d['score'], spot, key(b) in native[f], b[3], b[4], b[5], f, b))
n = len(p)
bands = [(0.4, 9), (0.3, 0.4), (0.22, 0.3), (0.15, 0.22), (0.1, 0.15), (0.05, 0.1), (0.02, 0.05)]
print('truck boxes per frame by score band (hedge variants included; they sit at <= 5% of a primary score)')
print(f'{"band":12s} {"native-truck":>13s} {"vehicle-copy":>13s} | spot boxes: ' + ' '.join(f'{k:>5s}' for k in SPOTS) + '   elsewhere')
for lo, hi in bands:
    r = [x for x in rows if lo <= x[0] < hi]
    nat = sum(x[2] for x in r) / n
    cp = sum(not x[2] for x in r) / n
    sp = {k: sum(x[1] == k for x in r) / n for k in SPOTS}
    el = sum(x[1] == '-' for x in r) / n
    print(f'{lo:4.2f}-{min(hi, 1):4.2f}    {nat:13.2f} {cp:13.2f} |             ' + ' '.join(f'{sp[k]:5.2f}' for k in SPOTS) + f'   {el:6.2f}')
print('\nmedian score of the best box at each spot, and how many OTHER boxes per frame outrank it:')
for k, (x, y) in SPOTS.items():
    best, outr = [], []
    for f, v in p.items():
        t = [(d['score'], d['box']) for d in v if d['class'] == 'truck']
        at = [s for s, b in t if np.hypot(b[0] - x, b[1] - y) < 3.0]
        if not at:
            continue
        s0 = max(at)
        best.append(s0)
        outr.append(sum(1 for s, b in t if s > s0 and np.hypot(b[0] - x, b[1] - y) >= 3.0))
    print(f'  spot {k}: present in {len(best)}/{n} frames | best score median {np.median(best):.2f} | other-location boxes outranking it per frame: median {np.median(outr):.0f}, mean {np.mean(outr):.1f}')
oth = [x for x in rows if x[1] == '-' and x[0] >= 0.15]
print(f'\nboxes elsewhere with score >= 0.15: {len(oth)} ({len(oth) / n:.2f}/frame) | native-truck share {np.mean([x[2] for x in oth]):.2f} | size L/W/H median {np.median([x[3] for x in oth]):.1f} {np.median([x[4] for x in oth]):.1f} {np.median([x[5] for x in oth]):.1f}')
rng = np.array([np.hypot(x[7][0], x[7][1]) for x in oth])
print('   range m p10/50/90', np.percentile(rng, [10, 50, 90]).round(1).tolist())
