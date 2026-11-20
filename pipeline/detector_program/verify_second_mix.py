"""Verify Claude's audited mixed dataset before its GPU slot."""
import argparse
import filecmp
import json
from pathlib import Path
import pickle
import re

import numpy as np


def ids(root, split):
    names = (root / f'ImageSets/{split}.txt').read_text().splitlines()
    if len(names) != len(set(names)) or any(not re.fullmatch(r'[\w.-]+', name) for name in names):
        raise ValueError(f'Duplicate or invalid frame IDs in {split}')
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ready', type=Path)
    parser.add_argument('base', type=Path)
    args = parser.parse_args()
    path = Path(json.loads(args.ready.read_text())['dataset_path'])
    if not path.is_absolute() or not path.is_dir():
        raise ValueError('Expected an absolute WSL dataset path')
    base_ids = set(ids(args.base, 'train'))
    validation_ids = ids(args.base, 'val')
    train_ids = ids(path, 'train')
    carla_ids = set(train_ids) - base_ids
    if (len(base_ids) != 5000 or len(train_ids) != 6000 or len(carla_ids) != 1000 or
            not base_ids <= set(train_ids) or ids(path, 'val') != validation_ids or
            set(validation_ids) & set(train_ids)):
        raise ValueError('Expected 5,000 preserved training frames, 1,000 new CARLA frames and the fixed 500 validation frames')
    for name in base_ids | set(validation_ids):
        for relative in (f'points/{name}.npy', f'labels/{name}.txt'):
            if not filecmp.cmp(args.base / relative, path / relative, shallow=False):
                raise ValueError(f'Frozen sample changed: {relative}')
    labels = set()
    for split, expected in (('train', train_ids), ('val', validation_ids)):
        infos = pickle.loads((path / f'custom_infos_{split}.pkl').read_bytes())
        if [row['point_cloud']['lidar_idx'] for row in infos] != expected:
            raise ValueError(f'Info file order differs from {split} IDs')
        for row in infos:
            name = row['point_cloud']['lidar_idx']
            ann = row['annos']
            boxes = ann['gt_boxes_lidar']
            classes = ann['name'].tolist()
            if boxes.shape != (len(classes), 7) or not np.isfinite(boxes).all() or (boxes[:, 3:6] <= 0).any():
                raise ValueError(f'Invalid boxes: {name}')
            lines = [line.split() for line in (path / 'labels' / f'{name}.txt').read_text().splitlines()]
            source_boxes = np.asarray([line[:-1] for line in lines], dtype=np.float32).reshape(-1, 7)
            if classes != [line[-1] for line in lines] or not np.array_equal(boxes, source_boxes):
                raise ValueError(f'Info boxes differ from labels: {name}')
            if name in carla_ids:
                if not set(classes) <= {'vehicle', 'pedestrian', 'truck'}:
                    raise ValueError(f'Unknown class: {name}')
                points = np.load(path / 'points' / f'{name}.npy', allow_pickle=False)
                if points.ndim != 2 or points.shape[1] != 4 or not len(points) or not np.isfinite(points).all():
                    raise ValueError(f'Invalid point cloud: {name}')
                labels.update(classes)
    if labels != {'vehicle', 'pedestrian', 'truck'}:
        raise ValueError('CARLA frames lack one or more classes')
    print(path)


if __name__ == '__main__':
    main()
