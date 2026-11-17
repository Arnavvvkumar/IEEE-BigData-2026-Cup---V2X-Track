"""Where is v5 still losing, and do the 50 test scans look like the validation scans?
 A) label-free comparison of v5 predictions on test vs dev: confident boxes per frame, their range, points inside them.
 B) on dev (labels): per class, by range band and by label point count: share of labels with a tight box (IoU>=0.5),
    a loose box (0.1<=IoU<0.5) or none, and AP restricted to the range band.
usage: gap_analysis.py <dev set dir> <norule pred name>"""
import importlib.util
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
F = W / 'experiments/claude_carla_overnight/final'
sys.path.insert(0, str(F))
import calib  # noqa: E402
import fuse_test as FT  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
C = ['vehicle', 'pedestrian', 'truck']
d, name = Path(sys.argv[1]), sys.argv[2]
RULE = {**calib.TRUCK, 'a': 0.5, 'base': 0.1}
dev = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **RULE))
test = json.loads((F / 'candidate_v5/predictions.json').read_text())
gt = {f: [g for g in v if g['points'] >= 5 and g.get('type') != 'TrashCan'] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
ids = sorted(gt)
BANDS = [(0, 15), (15, 25), (25, 35), (35, 200)]


def stats(pred, clouds, label):
    print(f'--- {label}: {len(pred)} frames')
    for c in C:
        rows = [(x['score'], np.hypot(*x['box'][:2]), x['box']) for f in pred for x in pred[f] if x['class'] == c]
        s = np.array([r[0] for r in rows]); r = np.array([r[1] for r in rows])
        out = f'  {c:10s} per frame: >=0.5 {(s >= .5).sum() / len(pred):5.2f}  >=0.3 {(s >= .3).sum() / len(pred):5.2f}  >=0.15 {(s >= .15).sum() / len(pred):5.2f}'
        m = s >= 0.3
        out += ' | range share of >=0.3 boxes: ' + ' '.join(f'{lo}-{hi}m {100 * ((r[m] >= lo) & (r[m] < hi)).mean():.0f}%' for lo, hi in BANDS)
        print(out)
    n = []
    for f in list(pred)[:50]:
        p = calib.nonground(np.asarray(clouds[f])[:, :3])
        for x in pred[f]:
            if x['class'] == 'pedestrian' and x['score'] >= 0.3:
                b = x['box']
                n.append(int(((np.abs(p[:, 0] - b[0]) < 0.5) & (np.abs(p[:, 1] - b[1]) < 0.5) & (np.abs(p[:, 2] - b[2]) < 1.1)).sum()))
    print(f'  points near confident pedestrians: p10/50/90 {np.percentile(n, [10, 50, 90]).round(0).tolist()}')
    tot = [len(calib.nonground(np.asarray(clouds[f])[:, :3])) for f in list(pred)[:50]]
    print(f'  non-ground points per scan: median {int(np.median(tot))}')


def band_ap(job):
    c, lo, hi = job
    inb = lambda b: lo <= np.hypot(b[0], b[1]) < hi
    p = {f: [x for x in dev[f] if x['class'] == c and inb(x['box'])] for f in ids}
    g = {f: [{'class': c, 'box': x['box']} for x in gt[f] if x['class'] == c and inb(x['box'])] for f in ids}
    n = sum(len(v) for v in g.values())
    if n < 20:
        return c, lo, hi, n, None
    m = ev.evaluate_3d_detection(p, g, [c], iou_thresholds=(0.5,), iou_type='3d')
    return c, lo, hi, n, round(100 * m['per_class'][c]['AP_0.5_R40'], 1)


if __name__ == '__main__':
    stats(test, FT.load(W / 'starting_kit/track_b/detection_test_frames'), 'TEST (v5 zip)')
    clouds = FT.load(d / 'scans')
    stats(dev, clouds, f'DEV {d.name} (v5 pipeline)')
    del clouds
    print('--- dev: label outcomes (best IoU with a same-class box of score >= 0.1)')
    rows = {c: [] for c in C}
    for f in ids:
        for c in C:
            pb = [np.array(x['box']) for x in dev[f] if x['class'] == c and x['score'] >= 0.1]
            for g in gt[f]:
                if g['class'] != c:
                    continue
                b = np.array(g['box'], float)
                near = [q for q in pb if np.hypot(q[0] - b[0], q[1] - b[1]) < 3]
                iou = max((ev._iou_3d(q, b) for q in near), default=0.0)
                rows[c].append((np.hypot(b[0], b[1]), g['points'], iou))
    for c in C:
        a = np.array(rows[c])
        print(f'  {c}: {len(a)} labels')
        for lab, col, bands in [('range m', 0, BANDS), ('label points', 1, [(5, 15), (15, 40), (40, 100), (100, 10 ** 6)])]:
            for lo, hi in bands:
                m = (a[:, col] >= lo) & (a[:, col] < hi)
                if m.sum() < 10:
                    continue
                i = a[m, 2]
                print(f'    {lab} {lo}-{hi}: n {int(m.sum()):5d} ({100 * m.mean():2.0f}%) tight {100 * (i >= .5).mean():3.0f}%  loose {100 * ((i >= .1) & (i < .5)).mean():3.0f}%  none {100 * (i < .1).mean():3.0f}%  median IoU {np.median(i):.2f}')
    print('--- dev: AP inside each range band')
    with get_context('fork').Pool(6) as pool:
        for c, lo, hi, n, v in pool.imap(band_ap, [(c, lo, hi) for c in C for lo, hi in BANDS]):
            print(f'  {c:10s} {lo:3d}-{hi:3d} m  labels {n:5d}  AP {v}', flush=True)
    print('GAP_DONE')
