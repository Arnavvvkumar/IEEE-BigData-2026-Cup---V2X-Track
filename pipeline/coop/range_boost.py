"""Test: far objects get systematically lower scores (fewer passes agree), so near false positives outrank far true positives.
Multiply the score of class c by 1 + k * clip((range - r0) / 20, 0, 1) and score AP (all / half A / half B).
usage: range_boost.py <set dir> <norule pred name> [workers]"""
import importlib.util
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
halves = [ids, ids[0::2], ids[1::2]]
gt = {f: [{'class': g['class'], 'box': g['box']} for g in G[f] if g['points'] >= 5] for f in ids}
pred = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))


def run(job):
    c, k, r0 = job
    p = {}
    for f in ids:
        out = []
        for x in pred[f]:
            if x['class'] == c:
                r = np.hypot(x['box'][0], x['box'][1])
                out.append({**x, 'score': x['score'] * (1 + k * float(np.clip((r - r0) / 20, 0, 1)))})
        p[f] = out
    res = []
    for h in halves:
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: gt[f] for f in h}, [c], iou_thresholds=(0.5,), iou_type='3d')
        res.append(round(100 * m['per_class'][c]['AP_0.5_R40'], 1))
    return f'{c:10s} k {k:4.1f} r0 {r0:2d}  AP {res}'


if __name__ == '__main__':
    jobs = [(c, k, r0) for c in ('vehicle', 'pedestrian') for k, r0 in [(0, 20), (0.3, 20), (0.6, 20), (1.0, 20), (1.5, 20), (0.6, 10), (1.0, 10), (-0.3, 20)]]
    with get_context('fork').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 5) as pool:
        for line in pool.imap(run, jobs):
            print(line, flush=True)
    print('RANGE_BOOST_DONE')
