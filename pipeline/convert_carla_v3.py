"""Convert carla_gen_v3.py clips (TILTED sensor) to the three-class OpenPCDet format.

Two outputs per frame:
  points/<id>.npy         LEVEL right-handed frame at the sensor position (yaw-only sensor frame): the exact known sensor
                          roll/pitch is removed, so boxes are computed exactly as in convert_carla.py (level-sensor rule).
                          Training later re-levels every frame with the points-only ground fit (canon/build_canon.py).
  tilted/<id>.npy         the native tilted sensor frame, right-handed, float32 xyz (for the realism scans).
Labels, classes, annotation rule, crop and intensity are identical to convert_carla.py. Synthetic data only."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

import convert_carla as C


def level_matrix(rec):
    """CARLA 4x4 matrix of the yaw-only transform at the sensor location."""
    x, y, z = rec['sensor_transform_carla']['location']
    yaw = math.radians(rec['sensor_transform_carla']['rotation'][2])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1.0]])


def level_points_rh(raw, rec):
    """Native sensor-local LH points -> world -> yaw-only sensor frame -> right-handed."""
    M = np.asarray(rec['sensor_matrix_carla'], dtype=np.float64)
    L = level_matrix(rec)
    p = np.c_[np.asarray(raw[:, :3], dtype=np.float64), np.ones(len(raw))] @ (np.linalg.inv(L) @ M).T
    out = np.zeros((len(raw), 4), dtype=np.float32)
    out[:, :3] = p[:, :3]
    out[:, 1] *= -1
    return out


def level_rec(rec):
    r = dict(rec)
    t = rec['sensor_transform_carla']
    r['sensor_transform_carla'] = {'location': t['location'], 'rotation': [0.0, 0.0, t['rotation'][2]]}
    r['sensor_matrix_carla'] = level_matrix(rec).tolist()
    return r


def convert_frame(raw, rec):
    pts = level_points_rh(raw, rec)
    lr = level_rec(rec)
    gt, details = [], []
    for a in rec['actors']:
        cls = C.classify(a['blueprint'])
        if cls is None:
            continue
        box = C.box_from_record(a, lr)
        hits = int(C.points_in_box(pts, box, zmin_offset=C.GROUND_SLAB).sum())
        cir = C.corners_in_range(box)
        ann = bool(hits >= C.MIN_POINTS and cir >= C.MIN_CORNERS)
        details.append({'id': a['id'], 'class': cls, 'blueprint': a['blueprint'], 'box': box.tolist(), 'hits': hits,
                        'corners_in_range': cir, 'annotated': ann, 'range_m': float(math.hypot(box[0], box[1]))})
        if ann:
            gt.append({'box': box.tolist(), 'class': cls})
    tilted = C.crop(C.native_points_rh(raw))[:, :3].astype(np.float32)
    return C.crop(pts).astype(np.float32), tilted, gt, details


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--native', type=Path, required=True, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prefix', default='carla3')
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(a.out)
    for d in ('points', 'labels', 'ImageSets', 'tilted'):
        (a.out / d).mkdir(parents=True)
    records, ids = [], []
    for ni, native in enumerate(a.native):
        for cdir in sorted(native.glob('clip_*')):
            if not (cdir / 'clip.json').exists():
                continue
            meta = json.loads((cdir / 'clip.json').read_text())
            for k, f in enumerate(meta['frames']):
                raw = np.load(io.BytesIO(C.read_exact(cdir / 'lidar' / f'{f:010d}.npy')))
                rec = json.loads((cdir / 'actors' / f'{f:010d}.json').read_text())
                pts, tilted, gt, det = convert_frame(raw, rec)
                fid = f"{a.prefix}_N{ni}_L{meta['clip']:04d}_F{k:02d}"
                np.save(a.out / 'points' / f'{fid}.npy', pts)
                np.save(a.out / 'tilted' / f'{fid}.npy', tilted)
                (a.out / 'labels' / f'{fid}.txt').write_text(''.join(
                    ' '.join(format(float(x), '.17g') for x in g['box']) + ' ' + g['class'] + '\n' for g in gt))
                records.append({'split': 'train', 'frame': fid, 'native': str(native), 'clip': meta['clip'], 'map': meta['map'],
                                'frame_in_clip': k, 'carla_frame': f, 'mount_height_m': meta['mount_height_m'],
                                'mount_rotation_roll_pitch_yaw': meta['mount_transform_carla']['rotation'], 'profile': meta['profile'],
                                'points': int(len(pts)), 'class_counts': {c: sum(g['class'] == c for g in gt) for c in ('vehicle', 'pedestrian', 'truck')},
                                'ground_truth': gt, 'object_details': det,
                                'native_lidar_sha256': C.sha(cdir / 'lidar' / f'{f:010d}.npy')})
                ids.append(fid)
    (a.out / 'ImageSets' / 'train.txt').write_text(''.join(i + '\n' for i in ids))
    (a.out / 'frames.json').write_text(json.dumps(records))
    tot = {c: sum(r['class_counts'][c] for r in records) for c in ('vehicle', 'pedestrian', 'truck')}
    prov = {'converter_sha256': C.sha(__file__), 'base_converter_sha256': C.sha(C.__file__), 'frames': len(ids),
            'clips': len({(r['native'], r['clip']) for r in records}), 'boxes': tot,
            'points_median': int(np.median([r['points'] for r in records])),
            'frames_json_sha256': hashlib.sha256((a.out / 'frames.json').read_bytes()).hexdigest()}
    (a.out / 'provenance.json').write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov))


if __name__ == '__main__':
    main()
