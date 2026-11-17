"""Independent audit of the 5k Generator V2 + 1k CARLA mix produced by build_mix.py (read-only)."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mix', type=Path, required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--carla', type=Path, required=True)
    ap.add_argument('--canonical-frames', type=Path, required=True)
    ap.add_argument('--expected-canonical-sha', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    mix, base, carla = a.mix, a.base, a.carla
    man = json.loads((mix / 'manifest.json').read_text())
    checks, notes = {}, {}
    base_train = (base / 'ImageSets/train.txt').read_text().split()
    base_val = (base / 'ImageSets/val.txt').read_text().split()
    carla_ids = (carla / 'ImageSets/train.txt').read_text().split()
    train = (mix / 'ImageSets/train.txt').read_text().split()
    val = (mix / 'ImageSets/val.txt').read_text().split()
    checks['train_is_base_then_carla'] = train == base_train + carla_ids
    checks['counts_5000_1000_500'] = (len(base_train), len(carla_ids), len(val), len(set(train))) == (5000, 1000, 500, 6000)
    checks['val_list_byte_identical'] = sha(mix / 'ImageSets/val.txt') == sha(base / 'ImageSets/val.txt') and val == base_val
    checks['no_val_in_train'] = not set(val) & set(train)
    checks['val_infos_identical'] = sha(mix / 'custom_infos_val.pkl') == sha(base / 'custom_infos_val.pkl')
    bad = [f for f in base_train + base_val for d, e in (('points', 'npy'), ('labels', 'txt'))
           if sha(mix / d / f'{f}.{e}') != sha(base / d / f'{f}.{e}')]
    checks['base_files_byte_identical'] = not bad
    rows = {r['id']: r for r in man['frames']}
    badc = [f for f in carla_ids if sha(mix / 'points' / f'{f}.npy') != sha(carla / 'points' / f'{f}.npy')
            or sha(mix / 'labels' / f'{f}.txt') != sha(carla / 'labels' / f'{f}.txt')
            or rows[f]['points_sha256'] != sha(carla / 'points' / f'{f}.npy')]
    checks['carla_files_match_conversion_and_manifest'] = not badc
    infos = pickle.loads((mix / 'custom_infos_train.pkl').read_bytes())
    binf = pickle.loads((base / 'custom_infos_train.pkl').read_bytes())
    same = len(infos) == 6000 and all(
        x['point_cloud'] == y['point_cloud'] and x['annos']['name'].tolist() == y['annos']['name'].tolist() and
        np.array_equal(x['annos']['gt_boxes_lidar'], y['annos']['gt_boxes_lidar']) for x, y in zip(infos[:5000], binf))
    checks['base_train_infos_unchanged'] = bool(same)
    cls, frames_with, pts, lab_bad = {'vehicle': 0, 'pedestrian': 0, 'truck': 0}, {'vehicle': 0, 'pedestrian': 0, 'truck': 0}, [], []
    for r in infos[5000:]:
        f = r['point_cloud']['lidar_idx']
        lines = [ln.split() for ln in (mix / 'labels' / f'{f}.txt').read_text().splitlines()]
        boxes = np.asarray([ln[:-1] for ln in lines], dtype=np.float32).reshape(-1, 7)
        names = [ln[-1] for ln in lines]
        if names != r['annos']['name'].tolist() or not np.array_equal(boxes, r['annos']['gt_boxes_lidar']) or \
                r['point_cloud']['num_features'] != 4:
            lab_bad.append(f)
        for c in set(names):
            frames_with[c] += 1
        for c in names:
            cls[c] += 1
        p = np.load(mix / 'points' / f'{f}.npy', allow_pickle=False)
        pts.append(len(p))
        if not (p.dtype == np.float32 and p.shape[1] == 4 and np.isfinite(p).all() and (np.abs(p[:, :2]) <= 40).all()
                and (p[:, 2] >= -8).all() and (p[:, 2] <= 2).all()):
            lab_bad.append(('cloud', f))
    checks['carla_infos_match_labels_and_clouds_valid'] = not lab_bad
    checks['carla_has_all_three_classes'] = all(v > 0 for v in cls.values())
    checks['canonical_frames_json_unchanged'] = sha(a.canonical_frames) == a.expected_canonical_sha
    checks['base_hashes_match_manifest'] = (sha(base / 'custom_infos_train.pkl') == man['base_train_infos_sha256'] and
                                            sha(base / 'provenance.json') == man['base_provenance_sha256'])
    notes.update(carla_boxes=cls, carla_frames_with_class=frames_with, carla_points_median=float(np.median(pts)),
                 bad_examples=(bad + badc + lab_bad)[:10])
    out = {'checks': checks, 'all_passed': all(checks.values()), 'notes': notes, 'audit_sha256': sha(__file__),
           'mix_train_infos_sha256': sha(mix / 'custom_infos_train.pkl'), 'manifest_sha256': sha(mix / 'manifest.json')}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
