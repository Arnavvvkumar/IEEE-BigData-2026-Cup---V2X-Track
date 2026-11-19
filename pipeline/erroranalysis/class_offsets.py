"""Diagnosis: systematic offsets of our vehicle / truck boxes vs real dev labels (radial, tangential, z, sizes, yaw).
usage: class_offsets.py <predictions.json>"""
import os
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
gt = json.loads((KIT / 'ground_truth.json').read_text())
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
pred = json.loads(Path(sys.argv[1]).read_text())


def mad(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


for c, rad in (('vehicle', 1.5), ('truck', 2.5), ('pedestrian', 0.6)):
    rows = []
    for f in ids:
        ps = sorted((d for d in pred[f] if d['class'] == c), key=lambda d: -d['score'])
        pb = np.array([d['box'] for d in ps]).reshape(-1, 7)
        for g in gt[f]:
            if g['class'] != c or not len(pb):
                continue
            b = np.array(g['box'], float)
            near = np.flatnonzero(np.hypot(pb[:, 0] - b[0], pb[:, 1] - b[1]) < rad)
            if not len(near):
                continue
            q = pb[near[0]]
            u = b[:2] / np.hypot(b[0], b[1])
            v = np.array([-u[1], u[0]])
            d = q[:3] - b[:3]
            dyaw = (q[6] - b[6] + np.pi / 2) % np.pi - np.pi / 2
            h = np.array([np.cos(b[6]), np.sin(b[6])])
            rows.append([d[:2] @ u, d[:2] @ v, d[2], q[3] - b[3], q[4] - b[4], q[5] - b[5], np.degrees(dyaw), d[:2] @ h,
                         d[:2] @ np.array([-h[1], h[0]]), (q[2] - q[5] / 2) - (b[2] - b[5] / 2)])
    a = np.array(rows)
    print(c, 'matched', len(a))
    for i, k in enumerate(('radial', 'tangential', 'z', 'dl', 'dw', 'dh', 'yaw_deg', 'along_heading', 'lateral', 'bottom')):
        print(f'  {k:14s} median={np.median(a[:, i]):+.3f} spread={mad(a[:, i]):.3f} p10={np.percentile(a[:, i], 10):+.3f} p90={np.percentile(a[:, i], 90):+.3f}')
