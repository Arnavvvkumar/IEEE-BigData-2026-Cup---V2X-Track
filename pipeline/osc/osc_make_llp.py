"""Runs ON OSC (login node, file operations only): build claude_twin_ll_v1p = the long-low frames that fully arrived (complete .npy and a
label file), with filtered train.txt / custom_infos_train.pkl, as symlinks into claude_twin_ll_v1. Prints counts."""
import os
import pickle
from pathlib import Path

import numpy as np

D = Path('/fs/scratch/' + os.environ['OSC_PROJECT'] + '/bdc/data')
src, out = D / 'claude_twin_ll_v1', D / 'claude_twin_ll_v1p'


def complete(p):
    try:
        with open(p, 'rb') as f:
            v = np.lib.format.read_magic(f)
            shape, _, dt = np.lib.format.read_array_header_1_0(f) if v == (1, 0) else np.lib.format.read_array_header_2_0(f)
            return os.path.getsize(p) == f.tell() + int(np.prod(shape)) * dt.itemsize
    except Exception:
        return False


ids = (src / 'ImageSets/train.txt').read_text().split()
infos = pickle.loads((src / 'custom_infos_train.pkl').read_bytes())
assert [i['point_cloud']['lidar_idx'] for i in infos] == ids
ok = [i for i in ids if (src / 'labels' / f'{i}.txt').exists() and complete(src / 'points' / f'{i}.npy')]
keep = set(ok)
if out.exists():
    raise SystemExit(f'{out} exists already')
for d in ('points', 'labels', 'ImageSets'):
    (out / d).mkdir(parents=True)
for i in ok:
    (out / 'points' / f'{i}.npy').symlink_to(src / 'points' / f'{i}.npy')
    (out / 'labels' / f'{i}.txt').symlink_to(src / 'labels' / f'{i}.txt')
(out / 'ImageSets/train.txt').write_text('\n'.join(ok) + '\n')
(out / 'custom_infos_train.pkl').write_bytes(pickle.dumps([x for x in infos if x['point_cloud']['lidar_idx'] in keep], protocol=4))
print('long-low frames: listed', len(ids), '| complete and usable', len(ok))
