"""End-to-end check that calib.ped_min_points (the build function) reproduces the dev gain of ped_pts_rank.py 'pts<5 x0.25'.
usage: verify_minpts.py <set dir> <norule pred name>"""
import os
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
gt = {f: [{'class': 'pedestrian', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
pred = json.loads((d / f'pred_{name}.json').read_text())
clouds = {f: np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
for label, p in (('base', pred), ('ped_min_points', calib.ped_min_points(pred, clouds))):
    q = calib.hedge_pedestrians({f: [x for x in p[f] if x['class'] == 'pedestrian'] for f in ids})
    out = []
    for h in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: q[f] for f in h}, {f: gt[f] for f in h}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 2))
    print(f'{d.name} {label:15s} pedestrian AP {out}', flush=True)
