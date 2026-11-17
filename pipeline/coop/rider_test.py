"""Test: rider-sized pedestrian boxes. ~5% of 'pedestrian' labels are bicycle / scooter riders 1.1-1.8 m long, while every
predicted box is ~0.75 m. If the non-ground points around a pedestrian box clearly span more than `thr` m along their main
horizontal axis (>= min_pts points), replace the box footprint by a rider box along that axis.
usage: rider_test.py <set dir> <pred name>"""
import importlib.util
import json
import sys
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
gt = {f: [{'class': g['class'], 'box': g['box']} for g in v if g['points'] >= 5 and g['class'] == 'pedestrian'] for f, v in G.items()}
ids = sorted(gt)
halves = [ids, ids[0::2], ids[1::2]]
pred = {f: [x for x in v if x['class'] == 'pedestrian'] for f, v in json.loads((d / f'pred_{name}.json').read_text()).items()}
clouds = FT.load(d / 'scans')
feat = {}
for f in ids:
    p = calib.nonground(np.asarray(clouds[f])[:, :3])
    rows = []
    for x in pred[f]:
        b = np.array(x['box'], float)
        q = p[(np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) < 1.1) & (np.abs(p[:, 2] - b[2]) <= b[5] / 2 + 0.1)][:, :2]
        if len(q) < 8:
            rows.append((0, 0, 0, 0, 0, 0))
            continue
        c = q.mean(0)
        w, v = np.linalg.eigh(np.cov((q - c).T))
        ax = v[:, 1]
        t, s = (q - c) @ ax, (q - c) @ v[:, 0]
        mid = c + ax * (t.max() + t.min()) / 2 + v[:, 0] * (s.max() + s.min()) / 2
        rows.append((np.ptp(t), np.ptp(s), len(q), mid[0], mid[1], np.arctan2(ax[1], ax[0])))
    feat[f] = rows
del clouds


def score(p):
    out = []
    for h in halves:
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: gt[f] for f in h}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 1))
    return out


def variant(job):
    if job is None:
        return f'base {score(pred)}'
    thr, mn, wmax, pad = job
    new, n = {}, 0
    for f in ids:
        out = []
        for x, (el, es, k, mx, my, yaw) in zip(pred[f], feat[f]):
            b = list(x['box'])
            if k >= mn and el >= thr and es <= wmax:
                b[0], b[1], b[3], b[4], b[6] = mx, my, float(np.clip(el + pad, 1.1, 1.9)), float(np.clip(es + pad, 0.6, 0.9)), yaw
                n += 1
            out.append({**x, 'box': b})
        new[f] = out
    return f'thr {thr} min_pts {mn} max_width {wmax} pad {pad}: changed {n} boxes {score(new)}'


if __name__ == '__main__':
    from multiprocessing import get_context
    jobs = [None, (1.2, 20, 0.9, 0.2), (1.4, 20, 0.9, 0.2), (1.2, 40, 0.9, 0.2), (1.4, 40, 0.8, 0.1), (1.6, 20, 0.9, 0.2)]
    with get_context('fork').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 6) as pool:  # forked workers share the loaded data
        for line in pool.imap(variant, jobs):
            print(line, flush=True)
    print('RIDER_DONE')
