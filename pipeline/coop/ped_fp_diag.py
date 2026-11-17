"""Where is pedestrian AP lost? (dev labels, analysis only). On the hedged v16 pedestrian predictions:
 - GT counts per frame with/without the >=5-point filter (official test: 953 pedestrians / 50 frames = 19.1 per frame)
 - replay the official greedy matching; classify every FALSE POSITIVE by what it overlaps: same GT already matched (duplicate),
   a pedestrian GT at IoU 0.1-0.5 (localisation), a <5-point pedestrian GT (filtered label), another class GT, or nothing (background)
 - for each class: count and score-weighted share among the top-ranked predictions (the part of the PR curve that matters)
 - recall ceiling, and AP if every FP of a type were removed (upper bound of fixing that type).
usage: ped_fp_diag.py <set dir> <norule pred name>"""
import importlib.util
import json
import sys
from collections import Counter
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
low = {f: [np.array(g['box'], float) for g in G[f] if g['points'] < 5 and g['class'] == 'pedestrian'] for f in ids}
oth = {f: [np.array(g['box'], float) for g in G[f] if g['class'] != 'pedestrian'] for f in ids}
allped = sum(1 for f in ids for g in G[f] if g['class'] == 'pedestrian')
print(f'{d.name}: frames {len(ids)} | pedestrian GT >=5 pts {sum(map(len, gt.values()))} ({sum(map(len, gt.values())) / len(ids):.1f}/frame) | '
      f'all {allped} ({allped / len(ids):.1f}/frame) | official test 19.1/frame')
pred = json.loads((d / f'pred_{name}.json').read_text())
p = calib.hedge_pedestrians({f: [x for x in pred[f] if x['class'] == 'pedestrian'] for f in ids})
rows = []   # (score, tp, type, frame)
for f in ids:
    used = [False] * len(gt[f])
    for x in sorted(p[f], key=lambda x: -x['score']):
        b = np.array(x['box'], float)
        ious = [ev._iou_3d(b, g) for g in gt[f]]
        bi, bj = 0.0, -1
        for j, i in enumerate(ious):
            if not used[j] and i > bi:
                bi, bj = i, j
        if bi >= 0.5:
            used[bj] = True; rows.append((x['score'], 1, 'TP', f)); continue
        best_any = max(ious) if ious else 0.0
        if best_any >= 0.5:
            t = 'duplicate'
        elif best_any >= 0.1:
            t = 'loc 0.1-0.5'
        elif any(ev._iou_3d(b, g) >= 0.1 for g in low[f]):
            t = 'low-point GT'
        elif any(np.hypot(*(b[:2] - g[:2])) < max(g[3], g[4]) / 2 + 0.3 for g in oth[f]):
            t = 'near vehicle/truck'
        else:
            t = 'background'
        rows.append((x['score'], 0, t, f))
rows.sort(key=lambda r: -r[0])
ngt = sum(map(len, gt.values()))
tp = np.cumsum([r[1] for r in rows])
print(f'recall ceiling {tp[-1] / ngt:.3f} | predictions {len(rows)}')
for k in (500, 1000, 2000, 3000, 5000):
    k = min(k, len(rows))
    c = Counter(r[2] for r in rows[:k])
    print(f'top {k:5d} (recall {tp[k - 1] / ngt:.3f}, precision {tp[k - 1] / k:.3f}): ' + ', '.join(f'{t} {n}' for t, n in c.most_common()))


def ap_of(rr):
    t = np.cumsum([r[1] for r in rr]); n = np.arange(1, len(rr) + 1)
    rec, prec = t / ngt, t / n
    s = 0.0
    for r in np.linspace(1 / 40, 1, 40):
        m = prec[rec >= r]
        s += m.max() if len(m) else 0.0
    return 100 * s / 40


print(f'AP (re-implemented R40) {ap_of(rows):.2f}')
for t in ('duplicate', 'loc 0.1-0.5', 'low-point GT', 'near vehicle/truck', 'background'):
    print(f'  AP if all "{t}" FPs were removed: {ap_of([r for r in rows if r[2] != t]):.2f}')
