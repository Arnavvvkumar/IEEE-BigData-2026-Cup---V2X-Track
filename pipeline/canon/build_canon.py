"""Ground-canonical copy of an OpenPCDet custom dataset: every frame (train and val) is levelled with the SAME points-only
ground fit used at test time (erroranalysis/groundplane.py) and shifted so the ground sits at --ground. Box centres follow the
same rigid transform; sizes and yaw are unchanged (synthetic scenes are near level). The source dataset is not modified.
Synthetic data only; no real data is read."""
import argparse
import hashlib
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'erroranalysis'))
from groundplane import fit_ground, level_transform  # noqa: E402

A = None


def read_labels(p):
    out = []
    for ln in Path(p).read_text().splitlines():
        *n, c = ln.split()
        out.append(([float(x) for x in n], c))
    return out


def work(job):
    i, f = job
    pts = np.load(A.src / 'points' / f'{f}.npy')
    try:
        abc = fit_ground(pts[:, :3], np.random.default_rng(i))
    except RuntimeError:
        return f, None, None
    R, t = level_transform(abc, A.ground)
    q = pts.copy()
    q[:, :3] = (pts[:, :3].astype(np.float64) @ R.T + t).astype(np.float32)
    np.save(A.out / 'points' / f'{f}.npy', q)
    L = read_labels(A.src / 'labels' / f'{f}.txt')
    lines, boxes = [], []
    for b, c in L:
        b = np.array(b, dtype=np.float64)
        b[:3] = R @ b[:3] + t
        boxes.append(b)
        lines.append(' '.join(format(float(x), '.17g') for x in b) + ' ' + c + '\n')
    (A.out / 'labels' / f'{f}.txt').write_text(''.join(lines))
    tilt = float(np.degrees(np.arctan(np.hypot(abc[0], abc[1]))))
    return f, [float(x) for x in abc] + [tilt], (np.array([c for _, c in L]), np.array(boxes, dtype=np.float32).reshape(-1, 7))


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ground', type=float, default=-3.0)
    ap.add_argument('--workers', type=int, default=12)
    A = ap.parse_args()
    if A.out.exists():
        raise FileExistsError(A.out)
    for d in ('points', 'labels', 'ImageSets'):
        (A.out / d).mkdir(parents=True)
    stats, failed = {}, []
    for split in ('train', 'val'):
        ids = (A.src / 'ImageSets' / f'{split}.txt').read_text().split()
        with Pool(A.workers) as pool:
            res = pool.map(work, list(enumerate(ids)), chunksize=16)
        keep, infos = [], []
        for f, st, an in res:
            if st is None:
                failed.append(f)
                continue
            keep.append(f)
            stats[f] = st
            infos.append({'point_cloud': {'num_features': 4, 'lidar_idx': f}, 'annos': {'name': an[0], 'gt_boxes_lidar': an[1]}})
        (A.out / 'ImageSets' / f'{split}.txt').write_text('\n'.join(keep) + '\n')
        (A.out / f'custom_infos_{split}.pkl').write_bytes(pickle.dumps(infos, protocol=4))
        print(split, len(keep), 'failed', len(ids) - len(keep), flush=True)
    S = np.array(list(stats.values()))
    v2 = np.array([v for k, v in stats.items() if not k.startswith('carla')])
    ca = np.array([v for k, v in stats.items() if k.startswith('carla')])
    man = {'source': str(A.src), 'target_ground_z': A.ground, 'frames': len(stats), 'failed': failed,
           'fit_c_percentiles_1_50_99': {'generator_v2': np.percentile(v2[:, 2], [1, 50, 99]).round(3).tolist() if len(v2) else None,
                                         'carla': np.percentile(ca[:, 2], [1, 50, 99]).round(3).tolist() if len(ca) else None},
           'fit_tilt_deg_percentiles_50_95_max': np.percentile(S[:, 3], [50, 95, 100]).round(3).tolist(),
           'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'train_infos_sha256': hashlib.sha256((A.out / 'custom_infos_train.pkl').read_bytes()).hexdigest()}
    (A.out / 'manifest.json').write_text(json.dumps(man, indent=1))
    (A.out / 'ground_fits.json').write_text(json.dumps(stats))
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
