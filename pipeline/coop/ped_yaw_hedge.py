"""Pedestrian hedge variants not tried yet, on the averaged-heading (FUSE_YAW_CIRC=vehicle,pedestrian) prediction:
rotated copies (+90 / +-45 deg) and height / z copies, added on top of the current PED_HEDGE. Pedestrian AP only, halves reported.
Pre-registered grid. usage: ped_yaw_hedge.py <set dir> <norule pred name> [workers]"""
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
gt = {f: [{'class': 'pedestrian', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
pred = {f: [x for x in v if x['class'] == 'pedestrian'] for f, v in json.loads((d / f'pred_{name}.json').read_text()).items()}


def extra(p, var, smin=0.05):
    """var: list of (dyaw rad, height scale, dz m, k) applied to primary boxes only"""
    out = {}
    for f, dets in p.items():
        new = []
        for x in dets:
            if x['score'] >= smin:
                b = x['box']
                for dyaw, hs, dz, k in var:
                    new.append({**x, 'box': [b[0], b[1], b[2] + dz, b[3], b[4], b[5] * hs, b[6] + dyaw], 'score': x['score'] * k})
        out[f] = new
    return out


def ap(p):
    out = []
    for h in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: gt[f] for f in h}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 2))
    return out


Q = np.pi / 4
JOBS = [
    ('current', []),
    ('+ yaw 90 f.1', [(2 * Q, 1, 0, 0.1)]),
    ('+ yaw 90 f.2', [(2 * Q, 1, 0, 0.2)]),
    ('+ yaw +-45 f.1', [(Q, 1, 0, 0.1), (-Q, 1, 0, 0.1)]),
    ('+ yaw 90,+-45 f.1', [(2 * Q, 1, 0, 0.1), (Q, 1, 0, 0.1), (-Q, 1, 0, 0.1)]),
    ('+ height .85/1.15 f.1', [(0, 0.85, 0, 0.1), (0, 1.15, 0, 0.1)]),
    ('+ z +-0.1 f.1', [(0, 1, 0.1, 0.1), (0, 1, -0.1, 0.1)]),
]
base = calib.hedge_pedestrians(pred)


def run(job):
    label, var = job
    e = extra(pred, var)
    p = {f: base[f] + e[f] for f in ids}
    return f'{d.name} {label:24s} ped AP {ap(p)} | boxes/frame {sum(len(v) for v in p.values()) / len(ids):.0f}'


if __name__ == '__main__':
    from multiprocessing import get_context
    with get_context('fork').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 2) as pool:
        for line in pool.imap(run, JOBS):
            print(line, flush=True)
    print('PED_YAW_DONE')
