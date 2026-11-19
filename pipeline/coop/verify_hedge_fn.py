"""End-to-end check that the function used in the build (calib.hedge_pedestrians) reproduces the validation gain.
usage: verify_hedge_fn.py <set dir> <norule pred name>"""
import os
import importlib.util
import json
import sys
from pathlib import Path

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
gt = {f: [{'class': 'pedestrian', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
pred = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))
for label, p in (('no hedging', pred), ('calib.hedge_pedestrians', calib.hedge_pedestrians(pred))):
    q = {f: [x for x in p[f] if x['class'] == 'pedestrian'] for f in ids}
    out = []
    for h in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: q[f] for f in h}, {f: gt[f] for f in h}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 1))
    print(f'{d.name} {name} {label:26s} pedestrian AP {out} boxes/frame {sum(len(v) for v in q.values()) / len(ids):.0f}', flush=True)
