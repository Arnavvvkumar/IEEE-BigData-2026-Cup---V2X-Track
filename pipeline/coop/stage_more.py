"""Stage the extra non-forbidden real frames as a validation set: xyz scans + labels in the roadside sensor frame.
Labels: union over agents by object id, centre inside the kit range, with raw type, source agent and number of sensor points.
usage: stage_more.py <fetched dir> <out dir>"""
import os
import json
import sys
from pathlib import Path

import numpy as np
import yaml

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'V2X-Real'))
from opencood.data_utils import SUPER_CLASS_MAP  # noqa: E402
from opencood.utils.transformation_utils import x1_to_x2  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / 'scans').mkdir(parents=True)
cls = {s: g for g, names in SUPER_CLASS_MAP.items() for s in names}
man = json.loads((src / 'manifest.json').read_text())
gt, index = {}, {}
for i, row in enumerate(sorted(man['frames'], key=lambda r: r['key'])):
    name = f'more_{i:03d}'
    files = row['files']
    pts = np.fromfile(src / files['-1.bin']['local'], dtype='<f4').reshape(-1, 4)
    xyz = pts[np.isfinite(pts[:, :3]).all(1), :3]
    xyz.tofile(out / 'scans' / f'{name}.bin')
    ego = yaml.safe_load((src / files['-1.yaml']['local']).read_text())
    objs = {k: (v, '-1') for k, v in ego['vehicles'].items()}
    for key, info in sorted(files.items()):
        if key.endswith('.yaml') and key != '-1.yaml':
            y = yaml.safe_load((src / info['local']).read_text())
            for k, v in (y.get('vehicles') or {}).items():
                objs.setdefault(k, (v, key[:-5]))
    p = xyz[np.abs(xyz).sum(1) > 0].astype(np.float64)
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
        truth.append({'class': c, 'box': [*centre.tolist(), *size.tolist(), yaw], 'agent': agent, 'points': n, 'type': obj['obj_type'], 'id': str(k)})
    gt[name] = truth
    index[name] = {'source_key': row['key'], 'scenario': row['key'].split('/')[0]}
(out / 'gt_coop.json').write_text(json.dumps(gt))
(out / 'index.json').write_text(json.dumps(index, indent=1))
F = len(gt)
print('frames', F, 'scenarios', len({v['scenario'] for v in index.values()}))
for label, keep in (('sensor-only', lambda g: g['agent'] == '-1'), ('coop >=5 pts', lambda g: g['points'] >= 5), ('coop all', lambda g: True)):
    print(f'  {label:13s}', {c: round(sum(g['class'] == c and keep(g) for v in gt.values() for g in v) / F, 2) for c in ('vehicle', 'pedestrian', 'truck')})
print('  test (organizers)', {'vehicle': 9.4, 'pedestrian': 19.06, 'truck': 2.48})
tc = {}
for v in gt.values():
    for g in v:
        if g['points'] >= 5:
            tc[f"{g['class'][:4]}/{g['type']}"] = tc.get(f"{g['class'][:4]}/{g['type']}", 0) + 1
print('  types (>=5 pts):', dict(sorted(tc.items(), key=lambda kv: -kv[1])))
