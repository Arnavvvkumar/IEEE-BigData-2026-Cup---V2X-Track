"""Vehicle fusion variants (CPU, existing predictions): cluster radius, box weighting, heading choice. Vehicle AP only.
Pipeline per variant: fuse (vehicle passes) -> calibrate -> AP on all / half A / half B.
usage: veh_fuse_test.py <set dir>"""
import os
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402
import fuse_test as FT  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d = Path(sys.argv[1])
gt = {f: [g for g in v if g['points'] >= 5] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
ids = sorted(gt)
halves = [ids, ids[0::2], ids[1::2]]
clouds = FT.load(d / 'scans')
TAGS = ['vnx010_all_s1234_ep10', 'vnx010_twin_s1_ep12', 'canon_all_s1234_ep12', 'second_twin_s1_ep12', 'canon_relabel_s1234_ep12', 'second_twin_s2_ep12']
passes = []
for t in TAGS:
    data = json.loads((d / 'allpred' / f'{t}_allpred.json').read_text())
    passes += [{f: [x for x in p[f] if x['class'] == 'vehicle'] for f in p} for p in data.values()]
n = len(passes)
print('vehicle passes', n, flush=True)


def fuse(radius, power, yaw_mode, iou_merge=None):
    out = {}
    for f in ids:
        items = sorted((x for p in passes for x in p[f]), key=lambda x: -x['score'])
        dets = []
        if items:
            B = np.array([x['box'] for x in items]); S = np.array([x['score'] for x in items])
            used = np.zeros(len(items), bool)
            for i in range(len(items)):
                if used[i]:
                    continue
                m = (~used) & (np.hypot(B[:, 0] - B[i, 0], B[:, 1] - B[i, 1]) < radius)
                used |= m
                w = S[m] ** power
                b = (B[m, :6] * w[:, None]).sum(0) / w.sum()
                if yaw_mode == 'top':
                    yaw = float(B[i, 6])
                else:  # weighted circular mean with period pi (a box and its 180-degree flip are the same box)
                    a = 2 * B[m, 6]
                    yaw = float(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum()) / 2)
                dets.append({'box': b.tolist() + [yaw], 'score': float(min(S[m].sum() / n, 1.0)), 'class': 'vehicle'})
        out[f] = dets
    return out


def ap(p):
    r = []
    for h in halves:
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: [{'class': x['class'], 'box': x['box']} for x in gt[f]] for f in h}, ['vehicle'], iou_thresholds=(0.5,), iou_type='3d')
        r.append(round(100 * m['per_class']['vehicle']['AP_0.5_R40'], 1))
    return r


for name, kw in [('base r0.6 p1 top', dict(radius=0.6, power=1, yaw_mode='top')),
                 ('r0.6 p1 circ-yaw', dict(radius=0.6, power=1, yaw_mode='circ')),
                 ('r0.6 p2 top', dict(radius=0.6, power=2, yaw_mode='top')),
                 ('r0.6 p2 circ-yaw', dict(radius=0.6, power=2, yaw_mode='circ')),
                 ('r0.9 p1 top', dict(radius=0.9, power=1, yaw_mode='top')),
                 ('r0.9 p2 circ-yaw', dict(radius=0.9, power=2, yaw_mode='circ')),
                 ('r1.2 p2 circ-yaw', dict(radius=1.2, power=2, yaw_mode='circ')),
                 ('r0.4 p1 top', dict(radius=0.4, power=1, yaw_mode='top'))]:
    print(f'{name:20s}', ap(calib.calibrate(fuse(**kw), clouds)), flush=True)
