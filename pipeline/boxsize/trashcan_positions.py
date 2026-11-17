"""Where are the 'TrashCan' boxes (selection half only)? Sensor-frame centres, sizes, yaw, and visible returns."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'V2X-Real'))
from opencood.utils.transformation_utils import x1_to_x2  # noqa: E402

KIT = ROOT / 'experiments/real_reference_audit/real_dev_cw1'
split = json.loads((ROOT / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json').read_text())
idx = json.loads((KIT / 'source_index.json').read_text())
rows = []
for kid in split['selection_ids']:
    lab = yaml.safe_load(Path(idx[kid]['label']).read_text())
    for oid, o in lab['vehicles'].items():
        if o['obj_type'] not in ('TrashCan', 'Truck', 'Bus', 'Van'):
            continue
        t = x1_to_x2([*(np.asarray(o['location']) + o['center']), *o['angle']], lab['lidar_pose'])
        c = t[:3, 3]
        yaw = np.degrees(np.arctan2(t[1, 0], t[0, 0]))
        pts = np.fromfile(KIT / 'detection_test_frames' / f'{kid}.bin', dtype='<f4').reshape(-1, 3)
        e = np.asarray(o['extent'])
        d = pts - c
        cy, sy = np.cos(np.radians(-yaw)), np.sin(np.radians(-yaw))
        lx, ly = cy * d[:, 0] - sy * d[:, 1], sy * d[:, 0] + cy * d[:, 1]
        n = int(((np.abs(lx) <= e[0]) & (np.abs(ly) <= e[1]) & (np.abs(d[:, 2]) <= e[2])).sum())
        rows.append((o['obj_type'], kid, idx[kid]['scenario'], oid, c.round(1).tolist(), (2 * e).round(2).tolist(), round(float(yaw), 1), n))
for r in sorted(rows):
    print(r)
