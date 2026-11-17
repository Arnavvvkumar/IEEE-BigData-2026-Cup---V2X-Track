"""Diagnostic first: at the same fused score, is a pedestrian box more or less likely to be a TP at different ranges? (If yes, the global ranking
is mis-ordered and a one-constant range factor could help.) Primary boxes after the v17 rule, greedy matching on primaries only.
Prints TP rate by score bin x range band, then (only if --rules) hedged AP for pre-registered range factors:
   s' = s * (1 - a * clip((r - 15) / 25, 0, 1)) for a in {0.2, 0.4}   (far discount)
   s' = s * (1 + a * clip((15 - r) / 15, 0, 1)) for a in {0.2, 0.4}   (near boost)
usage: ped_range_calib.py <set dir> <ped-only norule pred name> [rules]"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
gt = {f: [np.array(g['box'], float) for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
gtd = {f: [{'class': 'pedestrian', 'box': g.tolist()} for g in gt[f]] for f in ids}
clouds = {f: np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
pp = json.loads((d / f'pred_{name}.json').read_text())
P = calib.ped_min_points({f: [x for x in pp[f] if x['class'] == 'pedestrian'] for f in ids}, clouds)
rows = []
for f in ids:
    used = [False] * len(gt[f])
    for k in np.argsort([-x['score'] for x in P[f]]):
        b = np.array(P[f][k]['box'], float)
        bi, bj = 0.0, -1
        for j, g in enumerate(gt[f]):
            if not used[j]:
                i = ev._iou_3d(b, g)
                if i > bi: bi, bj = i, j
        if bi >= 0.5: used[bj] = True
        rows.append((P[f][k]['score'], float(np.hypot(b[0], b[1])), int(bi >= 0.5)))
a = np.array(rows)
top = a[np.argsort(-a[:, 0])[:6000]]
edges = np.quantile(top[:, 0], [0, 1 / 3, 2 / 3, 1.0])
bands = [(0, 15), (15, 25), (25, 80)]
print(f'{d.name} TP rate among the top-6000 primaries; score terciles (low -> high) x range band; n in brackets')
for lo, hi in zip(edges[:-1], edges[1:]):
    cells = []
    for r0, r1 in bands:
        m = (top[:, 0] >= lo) & (top[:, 0] <= hi) & (top[:, 1] >= r0) & (top[:, 1] < r1)
        cells.append(f'{r0}-{r1}m {top[m, 2].mean():.2f} [{m.sum()}]' if m.sum() else f'{r0}-{r1}m -')
    print(f'  score {lo:.2f}-{hi:.2f}: ' + ' | '.join(cells), flush=True)
if len(sys.argv) > 3:
    def ap(p):
        out = []
        for hh in (ids, ids[0::2], ids[1::2]):
            m = ev.evaluate_3d_detection({f: p[f] for f in hh}, {f: gtd[f] for f in hh}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
            out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 3))
        return out
    print(f'{d.name} v17 base ped AP {ap(calib.hedge_pedestrians(P))}', flush=True)
    for lab, fn in (('far discount a0.2', lambda r: 1 - 0.2 * np.clip((r - 15) / 25, 0, 1)), ('far discount a0.4', lambda r: 1 - 0.4 * np.clip((r - 15) / 25, 0, 1)),
                    ('near boost a0.2', lambda r: 1 + 0.2 * np.clip((15 - r) / 15, 0, 1)), ('near boost a0.4', lambda r: 1 + 0.4 * np.clip((15 - r) / 15, 0, 1))):
        Q = {f: [{**x, 'score': x['score'] * float(fn(np.hypot(x['box'][0], x['box'][1])))} for x in P[f]] for f in ids}
        print(f'{d.name} {lab:18s} ped AP {ap(calib.hedge_pedestrians(Q))}', flush=True)
print('RANGE_DONE')
