"""General truck post-processing variants on the v8 ensemble (v7 + PV-RCNN), checked on dev for NO HARM (dev cannot show gains on
the test clip's fixed long vehicles). Variants: long-box score boost (L >= 7 m), point-dense boost, truck clutter cut, truck hedging.
usage: truck_variants.py <set dir> <norule pred name> [workers]"""
import importlib.util
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402
import fuse_test as FT  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
halves = [ids, ids[0::2], ids[1::2]]
gt = {f: [g for g in G[f] if g['points'] >= 5 and g['class'] == 'truck'] for f in ids}
gen = {f: [g for g in gt[f] if g.get('type') != 'TrashCan'] for f in ids}
cans = {f: [np.array(g['box']) for g in gt[f] if g.get('type') == 'TrashCan'] for f in ids}
pred = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))
pred = {f: [x for x in pred[f] if x['class'] == 'truck'] for f in ids}
clouds = FT.load(d / 'scans')
npts = {}
for f in ids:
    p = calib.nonground(np.asarray(clouds[f])[:, :3])
    n = []
    for x in pred[f]:
        b = np.array(x['box'], float)
        q = p[np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) < max(b[3], b[4]) / 2 + 0.5] - b[:3]
        cs, sn = np.cos(-b[6]), np.sin(-b[6])
        lx, ly = q[:, 0] * cs - q[:, 1] * sn, q[:, 0] * sn + q[:, 1] * cs
        n.append(int(((np.abs(lx) <= b[3] / 2 + .2) & (np.abs(ly) <= b[4] / 2 + .2) & (np.abs(q[:, 2]) <= b[5] / 2 + .3)).sum()))
    npts[f] = n
del clouds


def ap(p, g):
    out = []
    for hh in halves:
        m = ev.evaluate_3d_detection({f: p[f] for f in hh}, {f: [{'class': 'truck', 'box': x['box']} for x in g[f]] for f in hh}, ['truck'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['truck']['AP_0.5_R40'], 1))
    return out


def variant(label):
    new = {}
    for f in ids:
        o = []
        for x, n in zip(pred[f], npts[f]):
            L, H, s = x['box'][3], x['box'][5], x['score']
            if label.startswith('long boost'):
                k = float(label.split('x')[1])
                s = min(1.0, s * k) if L >= 7.0 else s
            elif label.startswith('dense boost'):
                k = float(label.split('x')[1])
                s = min(1.0, s * k) if n >= 150 else s
            elif label.startswith('long+dense boost'):
                k = float(label.split('x')[1])
                s = min(1.0, s * k) if (L >= 7.0 and n >= 150) else s
            elif label.startswith('cut below'):
                if s < float(label.split()[-1]):
                    continue
            o.append({**x, 'score': s})
            if label == 'hedge length x height f.05' and x['score'] >= 0.02:
                b = x['box']
                for a in (.75, 1.3, 1.6):
                    for hs in (.85, 1.2, 1.4):
                        nh = b[5] * hs
                        o.append({**x, 'box': [b[0], b[1], b[2] - b[5] / 2 + nh / 2, b[3] * a, b[4], nh, b[6]], 'score': x['score'] * 0.05})
        new[f] = o
    pg = {f: [x for x in new[f] if not any(ev._iou_3d(np.array(x['box']), c) >= 0.3 for c in cans[f])] for f in ids}
    n = sum(len(v) for v in new.values()) / len(ids)
    return f'{label:28s} boxes/frame {n:6.0f}  truck {ap(new, gt)}  genuine {ap(pg, gen)}'


if __name__ == '__main__':
    jobs = ['base', 'long boost x1.5', 'long boost x2', 'dense boost x1.5', 'dense boost x2', 'long+dense boost x2',
            'cut below 0.03', 'cut below 0.08', 'hedge length x height f.05']
    with get_context('fork').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 3) as pool:
        for line in pool.imap(variant, jobs):
            print(line, flush=True)
    print('TRUCK_VARIANTS_DONE')
