"""Occupied-voxel audit for a candidate voxel size over every frame of one or more OpenPCDet custom datasets (read-only)."""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PCR = np.array([-40.0, -40.0, -8.0, 40.0, 40.0, 2.0])


def count(args):
    path, vs = args
    p = np.load(path, allow_pickle=False)[:, :3]
    m = np.all((p >= PCR[:3]) & (p < PCR[3:]), axis=1)
    idx = np.floor((p[m] - PCR[:3]) / vs).astype(np.int64)
    key = (idx[:, 0] * 2000 + idx[:, 1]) * 200 + idx[:, 2]
    return path.stem, int(len(np.unique(key))), int(m.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, nargs='+', required=True)
    ap.add_argument('--voxel', type=float, nargs=3, default=[0.05, 0.05, 0.25])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    vs = np.array(a.voxel)
    res = {}
    for d in a.data:
        ids = []
        for s in ('train', 'val'):
            f = d / 'ImageSets' / f'{s}.txt'
            if f.exists():
                ids += f.read_text().split()
        ids = sorted(set(ids))
        with Pool(a.workers) as pool:
            rows = pool.map(count, [(d / 'points' / f'{i}.npy', vs) for i in ids], chunksize=64)
        v = np.array([r[1] for r in rows])
        n = np.array([r[2] for r in rows])
        res[str(d)] = {'frames': len(rows), 'voxels': {'median': float(np.median(v)), 'p99': float(np.percentile(v, 99)), 'max': int(v.max()),
                                                     'argmax': rows[int(v.argmax())][0]},
                       'points_in_range': {'median': float(np.median(n)), 'max': int(n.max())},
                       'frames_over': {str(c): int((v > c).sum()) for c in (16000, 40000, 60000, 80000)}}
    out = {'voxel_size': a.voxel, 'range': PCR.tolist(), 'datasets': res}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
