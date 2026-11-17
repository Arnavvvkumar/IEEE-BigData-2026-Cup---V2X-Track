"""Build the 5,000 Generator V2 + 1,000 CARLA OpenPCDet training mix (fixed 500 synthetic validation preserved verbatim).

Base = data/codex_openpcdet_custom_v1 (frozen Generator V2: 5,000 train + 500 val). Nothing in it is modified.
  points/, labels/   per-file symlinks into the base for all 5,500 base frames; real copies for the 1,000 CARLA frames.
  ImageSets/train.txt  base train order, then the CARLA IDs in clip/frame order (6,000 unique).
  ImageSets/val.txt    byte-identical to the base val list.
  custom_infos_train.pkl  base train infos unchanged + CARLA infos built exactly like create_openpcdet_infos.py.
  custom_infos_val.pkl    symlink to the base val infos.
  manifest.json      every train/val ID with source, file sha256s, and the input hashes.
"""
import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def info_for(root, frame):
    names, boxes = [], []
    for line in (root / 'labels' / f'{frame}.txt').read_text().splitlines():
        *numbers, cls = line.split()
        names.append(cls)
        boxes.append([float(x) for x in numbers])
    return {'point_cloud': {'num_features': 4, 'lidar_idx': frame},
            'annos': {'name': np.array(names), 'gt_boxes_lidar': np.array(boxes, dtype=np.float32).reshape(-1, 7)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--carla', type=Path, required=True, help='convert_carla.py output (1,000 frames)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--expect-carla', type=int, default=1000)
    a = ap.parse_args(argv)
    base, carla, out = a.base.resolve(), a.carla.resolve(), a.out
    if out.exists():
        raise FileExistsError(out)
    base_train = (base / 'ImageSets/train.txt').read_text().split()
    base_val = (base / 'ImageSets/val.txt').read_text().split()
    carla_ids = (carla / 'ImageSets/train.txt').read_text().split()
    if len(base_train) != 5000 or len(base_val) != 500 or len(set(base_train) | set(base_val)) != 5500:
        raise ValueError('unexpected base splits')
    if len(carla_ids) != a.expect_carla or len(set(carla_ids)) != len(carla_ids) or set(carla_ids) & (set(base_train) | set(base_val)):
        raise ValueError('CARLA IDs missing, duplicated or colliding with base IDs')
    base_train_infos = pickle.loads((base / 'custom_infos_train.pkl').read_bytes())
    if [r['point_cloud']['lidar_idx'] for r in base_train_infos] != base_train:
        raise ValueError('base train infos do not follow base train order')
    for d in ('points', 'labels', 'ImageSets'):
        (out / d).mkdir(parents=True)
    for f in base_train + base_val:
        for d, ext in (('points', 'npy'), ('labels', 'txt')):
            (out / d / f'{f}.{ext}').symlink_to(base / d / f'{f}.{ext}')
    carla_rows = []
    for f in carla_ids:
        for d, ext in (('points', 'npy'), ('labels', 'txt')):
            shutil.copyfile(carla / d / f'{f}.{ext}', out / d / f'{f}.{ext}')
        pts = np.load(out / 'points' / f'{f}.npy', allow_pickle=False)
        if pts.dtype != np.float32 or pts.ndim != 2 or pts.shape[1] != 4 or not len(pts) or not np.isfinite(pts).all():
            raise ValueError(f'bad CARLA cloud {f}')
        carla_rows.append({'id': f, 'source': 'carla', 'points_sha256': sha(out / 'points' / f'{f}.npy'),
                           'labels_sha256': sha(out / 'labels' / f'{f}.txt'), 'num_points': int(len(pts))})
    train_ids = base_train + carla_ids
    (out / 'ImageSets/train.txt').write_text('\n'.join(train_ids) + '\n')
    shutil.copyfile(base / 'ImageSets/val.txt', out / 'ImageSets/val.txt')
    infos = base_train_infos + [info_for(out, f) for f in carla_ids]
    (out / 'custom_infos_train.pkl').write_bytes(pickle.dumps(infos, protocol=4))
    (out / 'custom_infos_val.pkl').symlink_to(base / 'custom_infos_val.pkl')
    base_rows = [{'id': f, 'source': 'generator_v2', 'split': s,
                  'points_sha256': sha(base / 'points' / f'{f}.npy'), 'labels_sha256': sha(base / 'labels' / f'{f}.txt')}
                 for s, ids in (('train', base_train), ('val', base_val)) for f in ids]
    counts = {c: 0 for c in ('vehicle', 'pedestrian', 'truck')}
    for r in infos[len(base_train):]:
        for n in r['annos']['name'].tolist():
            counts[n] += 1
    manifest = {
        'scope': '5,000 Generator V2 + 1,000 CARLA Town02 (64-channel level) training draws; fixed 500 Generator V2 validation',
        'base_path': str(base), 'carla_path': str(carla),
        'base_provenance_sha256': sha(base / 'provenance.json'),
        'base_train_infos_sha256': sha(base / 'custom_infos_train.pkl'), 'base_val_infos_sha256': sha(base / 'custom_infos_val.pkl'),
        'base_train_list_sha256': sha(base / 'ImageSets/train.txt'), 'base_val_list_sha256': sha(base / 'ImageSets/val.txt'),
        'carla_frames_json_sha256': sha(carla / 'frames.json'), 'carla_provenance_sha256': sha(carla / 'provenance.json'),
        'train_list_sha256': sha(out / 'ImageSets/train.txt'), 'val_list_sha256': sha(out / 'ImageSets/val.txt'),
        'train_infos_sha256': sha(out / 'custom_infos_train.pkl'),
        'counts': {'train': len(train_ids), 'generator_v2_train': len(base_train), 'carla_train': len(carla_ids), 'val': len(base_val)},
        'carla_boxes': counts, 'builder_sha256': sha(__file__),
        'frames': base_rows + carla_rows}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ('counts', 'carla_boxes', 'train_list_sha256', 'train_infos_sha256')}))


if __name__ == '__main__':
    main()
