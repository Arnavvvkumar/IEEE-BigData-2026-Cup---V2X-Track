"""Compare truck / vehicle box sizes: synthetic training set vs real validation labels by raw type (box-size statistics).
usage: size_compare.py <synthetic dataset dir> <real set dir> [more real set dirs]"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

syn = Path(sys.argv[1])
infos = pickle.loads((syn / 'custom_infos_train.pkl').read_bytes())
names = np.concatenate([i['annos']['name'] for i in infos])
boxes = np.concatenate([i['annos']['gt_boxes_lidar'].reshape(-1, 7) for i in infos])
bins = [0, 4, 4.5, 5, 5.5, 6, 7, 8, 10, 12, 20]


def show(tag, b):
    b = np.asarray(b).reshape(-1, 7) if np.asarray(b).shape[-1] == 7 else np.asarray(b)
    L, Wd, H = b[:, -4] if b.shape[1] == 7 else b[:, 0], b[:, -3] if b.shape[1] == 7 else b[:, 1], b[:, -2] if b.shape[1] == 7 else b[:, 2]
    h = np.histogram(L, bins)[0]
    print(f'{tag:28s} n={len(b):6d} median lwh {np.median(L):5.2f} {np.median(Wd):4.2f} {np.median(H):4.2f} | length share by bin {bins}: ' + ' '.join(f'{x / len(b):.2f}' for x in h))


print('SYNTHETIC', syn.name, 'frames', len(infos))
for c in ('vehicle', 'truck'):
    m = names == c
    show(c, boxes[m][:, 3:6])
print('REAL (>=5 sensor points)')
rows = {}
for d in sys.argv[2:]:
    for v in json.loads((Path(d) / 'gt_coop.json').read_text()).values():
        for g in v:
            if g['points'] >= 5 and g['class'] in ('vehicle', 'truck'):
                rows.setdefault(f"{g['class']}/{g['type']}", []).append(g['box'][3:6])
for k, v in sorted(rows.items(), key=lambda kv: -len(kv[1])):
    show(k, np.array(v))
