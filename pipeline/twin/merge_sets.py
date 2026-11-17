"""Merge OpenPCDet custom datasets by symlink: train = union of all inputs' train; val = the FIRST input's val (unchanged).
usage: merge_sets.py <out dir> <dataset A> <dataset B> ..."""
import hashlib
import json
import pickle
import sys
from pathlib import Path

out = Path(sys.argv[1])
srcs = [Path(s) for s in sys.argv[2:]]
if out.exists():
    raise FileExistsError(out)
for d in ('points', 'labels', 'ImageSets'):
    (out / d).mkdir(parents=True)
train_ids, train_infos = [], []
for s in srcs:
    ids = (s / 'ImageSets/train.txt').read_text().split()
    infos = pickle.loads((s / 'custom_infos_train.pkl').read_bytes())
    assert [i['point_cloud']['lidar_idx'] for i in infos] == ids, s
    train_ids += ids
    train_infos += infos
    for f in ids:
        (out / 'points' / f'{f}.npy').symlink_to(s / 'points' / f'{f}.npy')
        (out / 'labels' / f'{f}.txt').symlink_to(s / 'labels' / f'{f}.txt')
assert len(set(train_ids)) == len(train_ids)
a = srcs[0]
val_ids = (a / 'ImageSets/val.txt').read_text().split()
for f in val_ids:
    (out / 'points' / f'{f}.npy').symlink_to(a / 'points' / f'{f}.npy')
    (out / 'labels' / f'{f}.txt').symlink_to(a / 'labels' / f'{f}.txt')
(out / 'ImageSets/train.txt').write_text('\n'.join(train_ids) + '\n')
(out / 'ImageSets/val.txt').write_text('\n'.join(val_ids) + '\n')
(out / 'custom_infos_train.pkl').write_bytes(pickle.dumps(train_infos, protocol=4))
(out / 'custom_infos_val.pkl').write_bytes((a / 'custom_infos_val.pkl').read_bytes())
man = {'sources': [str(s) for s in srcs], 'train': len(train_ids), 'val': len(val_ids),
       'train_infos_sha256': hashlib.sha256((out / 'custom_infos_train.pkl').read_bytes()).hexdigest()}
(out / 'manifest.json').write_text(json.dumps(man, indent=1))
print(json.dumps(man, indent=1))
