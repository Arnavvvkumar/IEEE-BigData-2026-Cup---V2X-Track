"""Refiner training samples from SYNTHETIC site-twin frames only: every synthetic pedestrian box with >= 5 points, jittered
J times (centre, height), with the points around the jittered box and the true offset as the target.
usage: make_data.py <out npz> <frames per set> <set> [<set> ...]"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

D = Path.home() / 'projects/bdc2026-track-b/data'
J = 3


def work(job):
    s, f, seed = job
    rng = np.random.default_rng(seed)
    p = np.load(D / s / 'points' / f'{f}.npy')[:, :3].astype(np.float64)
    X, A, Y = [], [], []
    for ln in (D / s / 'labels' / f'{f}.txt').read_text().splitlines():
        *n, c = ln.split()
        if c != 'pedestrian':
            continue
        b = np.array([float(x) for x in n])
        near = p[(np.abs(p[:, 0] - b[0]) < 1.6) & (np.abs(p[:, 1] - b[1]) < 1.6)]
        q = near - b[:3]
        cs, sn = np.cos(-b[6]), np.sin(-b[6])
        lx, ly = q[:, 0] * cs - q[:, 1] * sn, q[:, 0] * sn + q[:, 1] * cs
        if ((np.abs(lx) <= b[3] / 2) & (np.abs(ly) <= b[4] / 2) & (np.abs(q[:, 2]) <= b[5] / 2)).sum() < 5:
            continue
        for _ in range(J):
            jb = b.copy()
            jb[:2] += np.clip(rng.normal(0, 0.13, 2), -0.35, 0.35)
            jb[2] += float(np.clip(rng.normal(0, 0.07), -0.2, 0.2))
            jb[5] *= 1 + float(np.clip(rng.normal(0, 0.06), -0.15, 0.15))
            r = C.crop(near, jb, rng)
            if r is None:
                continue
            x, aux, u, _ = r
            dxy = b[:2] - jb[:2]
            X.append(x); A.append(aux)
            Y.append([dxy[0] * u[0] + dxy[1] * u[1], -dxy[0] * u[1] + dxy[1] * u[0], b[2] - jb[2], b[5] - jb[5]])
    return np.array(X, np.float32).reshape(-1, C.K, 3), np.array(A, np.float32).reshape(-1, 3), np.array(Y, np.float32).reshape(-1, 4)


if __name__ == '__main__':
    out, per = sys.argv[1], int(sys.argv[2])
    jobs = []
    for s in sys.argv[3:]:
        ids = (D / s / 'ImageSets/train.txt').read_text().split()
        ids = ids[::max(1, len(ids) // per)][:per]
        jobs += [(s, f, 900000 + len(jobs) + i) for i, f in enumerate(ids)]
    with Pool(6) as pool:
        res = pool.map(work, jobs, chunksize=20)
    X = np.concatenate([r[0] for r in res]); A = np.concatenate([r[1] for r in res]); Y = np.concatenate([r[2] for r in res])
    np.savez_compressed(out, X=X.astype(np.float16), A=A, Y=Y)
    print('frames', len(jobs), 'samples', len(X), '| target |dr| |dt| |dz| |dh| mean', np.abs(Y).mean(0).round(3).tolist())
