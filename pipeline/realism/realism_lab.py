"""CPU realism experiments with the kit's unmodified metric (seed 1234, 10,000 points) against the 50 public real dev scans.
Permitted use of the public reference: realism calibration of the simulator output (sensor pose, range falloff).
Steps: A level CARLA (as is)  B tilted to the real ground plane and sensor height  C + per-clip yaw search
       D + range-falloff thinning to the real radial point distribution.
Synthetic scans are CARLA 64-channel captures (claude_carla64_town02_v1); real points are only compared, never copied."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'starting_kit/track_b/scoring_program'))
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/erroranalysis'))
from metrics import compute_realism, compute_fpd  # noqa: E402
from groundplane import fit_ground, level_transform  # noqa: E402

KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
DATA = Path.home() / 'projects/bdc2026-track-b/data/claude_carla64_town02_v1'
RANGE = (-40.0, -40.0, -8.0, 40.0, 40.0, 2.0)
NORM = {'CD': (2.0, 30.0), 'MMD': (0.0, 0.5), 'EMD': (0.0, 5.0), 'FPD': (0.0, 50.0)}
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
real = [np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids]
planes = np.array([fit_ground(c, np.random.default_rng(i)) for i, c in enumerate(real)])
PLANE = np.median(planes, 0)
print('real plane', PLANE.round(4).tolist(), flush=True)


def mask(p):
    return p[(p[:, 0] >= RANGE[0]) & (p[:, 0] <= RANGE[3]) & (p[:, 1] >= RANGE[1]) & (p[:, 1] <= RANGE[4]) & (p[:, 2] >= RANGE[2]) & (p[:, 2] <= RANGE[5])]


def norm_score(raw):
    s = {k: float(np.clip((NORM[k][1] - raw[k]) / (NORM[k][1] - NORM[k][0]), 0, 1)) for k in NORM}
    return {'realism': round(float(np.mean(list(s.values()))), 4), **{k: round(raw[k], 4) for k in NORM}, **{k + '_n': round(v, 3) for k, v in s.items()}}


def lean_realism(syn, ref, seed=1234):
    """Same computation and RNG order as metrics.compute_realism, with the MMD kernel sums done in chunks (WSL has 7.5 GB).
    Final candidates are re-scored with the unmodified kit function when memory allows."""
    import metrics as M
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    Xs = M._subsample(M._concat_and_mask(syn, RANGE), M.N_SUBSAMPLE, rng)
    Xr = M._subsample(M._concat_and_mask(ref, RANGE), M.N_SUBSAMPLE, rng)
    cd = M.chamfer_distance(Xs, Xr)
    gamma = M._median_heuristic_gamma(Xs, Xr, rng=rng)
    n, m = len(Xs), len(Xr)
    sxx = syy = sxy = 0.0
    for i in range(0, n, 1000):
        a, b = Xs[i:i + 1000], Xr[i:i + 1000]
        sxx += np.exp(-gamma * cdist(a, Xs, 'sqeuclidean')).sum() - len(a)  # minus the unit diagonal
        syy += np.exp(-gamma * cdist(b, Xr, 'sqeuclidean')).sum() - len(b)
        sxy += np.exp(-gamma * cdist(a, Xr, 'sqeuclidean')).sum()
    mmd = sxx / (n * (n - 1)) + syy / (m * (m - 1)) - 2.0 * sxy / (n * m)
    return {'CD': float(cd), 'MMD': float(mmd), 'EMD': M.compute_emd(Xs, Xr), 'FPD': M.compute_fpd(Xs, Xr)}


def full(clouds):
    return norm_score(lean_realism(clouds, real))


real_pool = mask(np.concatenate(real))
rs = real_pool[np.random.default_rng(0).choice(len(real_pool), 20000, replace=False)]
tree_r = cKDTree(rs)


def proxy(pool, rng):
    """Fast stand-in for the composite: normalised CD + FPD on 20k points (lower is better)."""
    s = pool[rng.choice(len(pool), 20000, replace=len(pool) < 20000)]
    cd = float(np.mean(tree_r.query(s)[0] ** 2) + np.mean(cKDTree(s).query(rs)[0] ** 2))
    return cd / 28 + compute_fpd(s, rs) / 50


def to_real_frame(p, ground_c, yaw):
    """Level synthetic scan (ground at z = ground_c) -> rotate about z by yaw -> tilt/shift onto the real ground plane."""
    c, s = np.cos(yaw), np.sin(yaw)
    q = p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    R, t = level_transform(PLANE, ground_c)  # real -> level at ground_c; invert it
    return ((q - t) @ R).astype(np.float32)


frames = json.loads((DATA / 'frames.json').read_text())
train_ids = (DATA / 'ImageSets/train.txt').read_text().split()
clips = {}
for f in train_ids:
    clips.setdefault(f.rsplit('_', 1)[0], []).append(f)
names = sorted(clips)
print('clips', len(names), 'frames per clip', len(clips[names[0]]), flush=True)


def load(f):
    return np.load(DATA / 'points' / f'{f}.npy')[:, :3].astype(np.float64)


rng = np.random.default_rng(1)
out = {}
# A: level, as generated: one frame from each of 20 clips x 2-3 -> 50 frames spread over clips
pick = [clips[n][i] for i in (5, 20, 35) for n in names][:50]
level = [load(f) for f in pick]
gc = float(np.median([fit_ground(c, np.random.default_rng(i))[2] for i, c in enumerate(level[:10])]))
print('carla ground', round(gc, 3), flush=True)
out['A_level'] = full([c.astype(np.float32) for c in level])
print('A', out['A_level'], flush=True)
out['B_tilted'] = full([to_real_frame(c, gc, 0.0) for c in level])
print('B', out['B_tilted'], flush=True)

# C: yaw search per clip (24 angles) on the clip's middle frame; rank clips by proxy
YAWS = np.radians(np.arange(0, 360, 15))
best = {}
for n in names:
    c = load(clips[n][20])
    sc = [proxy(mask(to_real_frame(c, gc, y)), rng) for y in YAWS]
    best[n] = (float(YAWS[int(np.argmin(sc))]), float(min(sc)))
rank = sorted(names, key=lambda n: best[n][1])
print('clip proxies', [(n, round(np.degrees(best[n][0])), round(best[n][1], 3)) for n in rank], flush=True)


def assemble(use, per):
    fs = [(n, clips[n][i]) for n in use for i in np.linspace(0, len(clips[n]) - 1, per).astype(int)][:50]
    return fs, [to_real_frame(load(f), gc, best[n][0]) for n, f in fs]


for label, use, per in (('C_yaw_all25', names, 2), ('C_yaw_top10', rank[:10], 5), ('C_yaw_top5', rank[:5], 10), ('C_yaw_top2', rank[:2], 25)):
    fs, cl = assemble(use, per)
    out[label] = full(cl)
    print(label, out[label], flush=True)

# D: range-falloff thinning so the pooled radial distribution matches the real one (1 m bins)
edges = np.arange(0, 58, 1.0)
hr = np.histogram(np.hypot(real_pool[:, 0], real_pool[:, 1]), edges)[0] / len(real_pool)
for label, use, per in (('D_thin_top10', rank[:10], 5), ('D_thin_top5', rank[:5], 10), ('D_thin_all25', names, 2)):
    fs, cl = assemble(use, per)
    pool = mask(np.concatenate(cl))
    hs = np.histogram(np.hypot(pool[:, 0], pool[:, 1]), edges)[0] / len(pool)
    w = np.where(hs > 0, hr / np.maximum(hs, 1e-9), 0)
    w = w / w.max()
    thin = []
    for c in cl:
        b = np.clip(np.digitize(np.hypot(c[:, 0], c[:, 1]), edges) - 1, 0, len(w) - 1)
        thin.append(c[rng.random(len(c)) < w[b]])
    out[label] = full(thin)
    out[label]['points_per_scan_median'] = int(np.median([len(mask(c)) for c in thin]))
    print(label, out[label], flush=True)
out['meta'] = {'real_plane_abc': PLANE.tolist(), 'carla_ground': gc, 'clip_best_yaw_deg_proxy': {n: [round(np.degrees(best[n][0])), round(best[n][1], 4)] for n in names}}
Path(__file__).with_name('realism_lab_v1.json').write_text(json.dumps(out, indent=1))
