"""Cooperative dev labels: union (by object id) of every agent's labels for each dev frame, expressed in the roadside sensor
(-1) frame, centre inside the kit range. Compares box counts with the organizers' test labels and scores a prediction file
against both label sets. usage: build_coop_gt.py <labels dir> <out gt.json> [predictions.json]"""
import os
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'V2X-Real'))
from opencood.data_utils import SUPER_CLASS_MAP  # noqa: E402
from opencood.utils.transformation_utils import x1_to_x2  # noqa: E402

KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
index = json.loads((KIT / 'source_index.json').read_text())
old = json.loads((KIT / 'ground_truth.json').read_text())
cls = {s: g for g, names in SUPER_CLASS_MAP.items() for s in names}
C = ['vehicle', 'pedestrian', 'truck']
labels = Path(sys.argv[1])
gt, stats = {}, {'agents': {}, 'added_by_other_agents': {c: 0 for c in C}, 'points_in_added': {c: [] for c in C}}
for name, row in index.items():
    ego = yaml.safe_load(Path(row['label']).read_text())
    scenario, _, stem = row['source_key'].split('/')
    objs = {k: (v, '-1') for k, v in ego['vehicles'].items()}
    for other in sorted(labels.glob(f'{scenario}/*/{stem}.yaml')):
        y = yaml.safe_load(other.read_text())
        stats['agents'][other.parent.name] = stats['agents'].get(other.parent.name, 0) + 1
        for k, v in (y.get('vehicles') or {}).items():
            objs.setdefault(k, (v, other.parent.name))
    p = np.fromfile(KIT / 'detection_test_frames' / f'{name}.bin', dtype='<f4').reshape(-1, 3).astype(np.float64)
    p = p[np.abs(p).sum(1) > 0]
    truth = []
    for k, (obj, agent) in objs.items():
        c = cls.get(obj['obj_type'])
        if c is None:
            continue
        pose = [*(np.asarray(obj['location']) + obj['center']), *obj['angle']]
        T = x1_to_x2(pose, ego['lidar_pose'])
        centre = T[:3, 3]
        if (centre < [-40, -40, -8]).any() or (centre > [40, 40, 2]).any():
            continue
        size = 2 * np.asarray(obj['extent'])
        yaw = float(np.arctan2(T[1, 0], T[0, 0]))
        q = p[np.abs(p - centre).max(1) < size.max()] - centre
        cs, sn = np.cos(-yaw), np.sin(-yaw)
        x = q[:, 0] * cs - q[:, 1] * sn
        yy = q[:, 0] * sn + q[:, 1] * cs
        n = int(((np.abs(x) <= size[0] / 2) & (np.abs(yy) <= size[1] / 2) & (np.abs(q[:, 2]) <= size[2] / 2)).sum())
        truth.append({'class': c, 'box': [*centre.tolist(), *size.tolist(), yaw], 'agent': agent, 'points': n, 'type': obj['obj_type']})
        if agent != '-1':
            stats['added_by_other_agents'][c] += 1
            stats['points_in_added'][c].append(n)
    gt[name] = truth
Path(sys.argv[2]).write_text(json.dumps(gt))
F = len(gt)
print('agent label files used', stats['agents'])
print('per frame          vehicle pedestrian truck')
print('  test (organizers)   9.40      19.06  2.48')
print('  sensor-only dev  ', ' '.join(f"{sum(g['class'] == c for v in old.values() for g in v) / F:9.2f}" for c in C))
print('  cooperative dev  ', ' '.join(f"{sum(g['class'] == c for v in gt.values() for g in v) / F:9.2f}" for c in C))
for c in C:
    n = np.array(stats['points_in_added'][c])
    if len(n):
        print(f'  added {c}: {len(n)} boxes; sensor points inside: 0 pts {int((n == 0).sum())}, 1-4 {int(((n > 0) & (n < 5)).sum())}, 5-19 {int(((n >= 5) & (n < 20)).sum())}, 20+ {int((n >= 20).sum())}')
if len(sys.argv) > 3:
    spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    pred = json.loads(Path(sys.argv[3]).read_text())
    for nm, g in (('sensor-only labels', old), ('cooperative labels', gt)):
        m = ev.evaluate_3d_detection(pred, {f: [{'class': x['class'], 'box': x['box']} for x in v] for f, v in g.items()}, C, iou_thresholds=(0.5,), iou_type='3d')
        print(nm, 'mAP', round(100 * m['mAP_0.5_R40'], 2), {c: round(100 * m['per_class'][c]['AP_0.5_R40'], 2) for c in C})
    print('real test result      mAP 49.33 {vehicle: 63.76, pedestrian: 31.30, truck: 52.92}')
