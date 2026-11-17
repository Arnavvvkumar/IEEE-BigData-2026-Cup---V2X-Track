"""Audit of our dev label conversion vs what the organizers' test labels look like (test: 470 vehicle / 953 pedestrian /
124 truck boxes on 50 frames = 9.4 / 19.1 / 2.5 per frame). Counts raw V2X-Real objects in the 50 dev frames by type, by
range band and by number of lidar points inside, to see which filtering rule reproduces the test counts."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'V2X-Real'))
from opencood.data_utils import SUPER_CLASS_MAP  # noqa: E402
from opencood.utils.transformation_utils import x1_to_x2  # noqa: E402

KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
index = json.loads((KIT / 'source_index.json').read_text())
print('SUPER_CLASS_MAP', json.dumps(SUPER_CLASS_MAP))
cls = {s: g for g, names in SUPER_CLASS_MAP.items() for s in names}
types, rows = Counter(), []
for name, row in index.items():
    frame = yaml.safe_load(Path(row['label']).read_text())
    p = np.fromfile(KIT / 'detection_test_frames' / f'{name}.bin', dtype='<f4').reshape(-1, 3).astype(np.float64)
    p = p[np.abs(p).sum(1) > 0]
    for obj in frame['vehicles'].values():
        pose = [*(np.asarray(obj['location']) + obj['center']), *obj['angle']]
        T = x1_to_x2(pose, frame['lidar_pose'])
        c = T[:3, 3]
        size = 2 * np.asarray(obj['extent'])
        yaw = np.arctan2(T[1, 0], T[0, 0])
        q = p[np.abs(p - c).max(1) < size.max()] - c
        cs, sn = np.cos(-yaw), np.sin(-yaw)
        x = q[:, 0] * cs - q[:, 1] * sn
        y = q[:, 0] * sn + q[:, 1] * cs
        n = int(((np.abs(x) <= size[0] / 2) & (np.abs(y) <= size[1] / 2) & (np.abs(q[:, 2]) <= size[2] / 2)).sum())
        inr = bool((c >= [-40, -40, -8]).all() and (c <= [40, 40, 2]).all())
        types[(obj['obj_type'], cls.get(obj['obj_type']), inr)] += 1
        rows.append((cls.get(obj['obj_type']), obj['obj_type'], inr, max(abs(c[0]), abs(c[1])), c[2], n))
print('per raw type (type, superclass, centre in range): count')
for k, v in sorted(types.items(), key=lambda kv: -kv[1]):
    print('  ', k, v)
F = len(index)
print('per frame, test labels: vehicle 9.4 pedestrian 19.06 truck 2.48')
for g in ('vehicle', 'pedestrian', 'truck'):
    r = [x for x in rows if x[0] == g]
    inr = [x for x in r if x[2]]
    print(g, 'all', round(len(r) / F, 2), '| in range', round(len(inr) / F, 2), '| in range by points: 0 pts', sum(x[5] == 0 for x in inr),
          '1-4 pts', sum(1 <= x[5] < 5 for x in inr), '5+', sum(x[5] >= 5 for x in inr),
          '| out of range with 5+ pts', sum((not x[2]) and x[5] >= 5 for x in r), '| centre max|xy| bands 40-50/50-60/60-80/80+:',
          [sum((not x[2]) and lo <= x[3] < hi for x in r) for lo, hi in ((40, 50), (50, 60), (60, 80), (80, 1e9))],
          '| xy<40 but z out of [-8,2]:', sum((not x[2]) and x[3] < 40 for x in r))
