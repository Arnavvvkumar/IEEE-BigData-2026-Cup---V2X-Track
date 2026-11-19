"""Choose 50 realism scans from the sensor-matched CARLA v3 capture (native tilted frame, untouched points).
Greedy clip selection on a fast pooled proxy (CD + FPD + sliced Wasserstein), then the kit metric (chunked MMD, same maths)
on the assembled 50-scan sets. Calibration against the public real reference only; the scans themselves are pure simulator output."""
import os
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'starting_kit/track_b/scoring_program'))
import metrics as M  # noqa: E402

KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
DATA = Path.home() / 'projects/bdc2026-track-b/data/claude_carla_v3'
RANGE = (-40.0, -40.0, -8.0, 40.0, 40.0, 2.0)
NORM = {'CD': (2.0, 30.0), 'MMD': (0.0, 0.5), 'EMD': (0.0, 5.0), 'FPD': (0.0, 50.0)}
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
real = [np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids]


def lean(syn, ref, seed=1234):
    rng = np.random.default_rng(seed)
    Xs = M._subsample(M._concat_and_mask(syn, RANGE), M.N_SUBSAMPLE, rng)
    Xr = M._subsample(M._concat_and_mask(ref, RANGE), M.N_SUBSAMPLE, rng)
    cd = M.chamfer_distance(Xs, Xr)
    gamma = M._median_heuristic_gamma(Xs, Xr, rng=rng)
    n, m = len(Xs), len(Xr)
    sxx = syy = sxy = 0.0
    for i in range(0, n, 1000):
        a, b = Xs[i:i + 1000], Xr[i:i + 1000]
        sxx += np.exp(-gamma * cdist(a, Xs, 'sqeuclidean')).sum() - len(a)
        syy += np.exp(-gamma * cdist(b, Xr, 'sqeuclidean')).sum() - len(b)
        sxy += np.exp(-gamma * cdist(a, Xr, 'sqeuclidean')).sum()
    raw = {'CD': float(cd), 'MMD': float(sxx / (n * (n - 1)) + syy / (m * (m - 1)) - 2.0 * sxy / (n * m)),
           'EMD': M.compute_emd(Xs, Xr), 'FPD': M.compute_fpd(Xs, Xr)}
    s = {k: float(np.clip((NORM[k][1] - raw[k]) / (NORM[k][1] - NORM[k][0]), 0, 1)) for k in NORM}
    return {'realism': round(float(np.mean(list(s.values()))), 4), **{k: round(v, 4) for k, v in raw.items()}, **{k + '_n': round(v, 3) for k, v in s.items()}}


def mask(p):
    return M._mask_to_range(p, RANGE)


rng = np.random.default_rng(0)
rp = mask(np.concatenate(real))
rs = rp[rng.choice(len(rp), 20000, replace=False)]
tree_r = cKDTree(rs)
DIRS = rng.normal(size=(64, 3))
DIRS /= np.linalg.norm(DIRS, axis=1, keepdims=True)
rproj = np.sort(rs @ DIRS.T, axis=0)


def proxy(pool):
    s = pool[rng.choice(len(pool), 20000, replace=len(pool) < 20000)]
    cd = float(np.mean(tree_r.query(s)[0] ** 2) + np.mean(cKDTree(s).query(rs)[0] ** 2))
    sw = float(np.mean(np.abs(np.sort(s @ DIRS.T, axis=0) - rproj)))
    return cd / 28 + M.compute_fpd(s, rs) / 50 + sw / 5


frames = (DATA / 'ImageSets/train.txt').read_text().split()
clips = {}
for f in frames:
    clips.setdefault(f.rsplit('_F', 1)[0], []).append(f)
names = sorted(clips)


def load(f):
    return np.load(DATA / 'tilted' / f'{f}.npy')


mid = {n: mask(load(clips[n][20])) for n in names}
single = {n: proxy(mid[n]) for n in names}
rank = sorted(names, key=single.get)
print('best single clips', [(n, round(single[n], 3)) for n in rank[:8]], flush=True)
# greedy pooled selection (clips may repeat = more frames from that clip)
chosen, best = [], 1e9
for step in range(10):
    cand = []
    for n in rank[:40]:
        cand.append((proxy(np.concatenate([mid[c] for c in chosen + [n]])), n))
    sc, n = min(cand)
    if sc >= best - 1e-4 and len(chosen) >= 3:
        break
    chosen.append(n); best = sc
    print('greedy', step, n, round(sc, 4), flush=True)
out = {}


def assemble(use):
    per = int(np.ceil(50 / len(use)))
    fs = [clips[n][i] for n in use for i in np.linspace(0, 39, per).astype(int)][:50]
    return fs


for label, use in (('v3_random25', [names[i] for i in np.linspace(0, len(names) - 1, 25).astype(int)]),
                   ('v3_top10_single', rank[:10]), ('v3_top5_single', rank[:5]), ('v3_greedy', sorted(set(chosen), key=chosen.index))):
    fs = assemble(use)
    out[label] = {**lean([load(f) for f in fs], real), 'clips': use, 'frames': fs,
                  'points_per_scan_median': int(np.median([len(mask(load(f))) for f in fs]))}
    print(label, {k: v for k, v in out[label].items() if k not in ('frames', 'clips')}, flush=True)
# noise floor: real selection half vs real confirmation half (what a perfect generator would score against 25 scans)
split = json.loads((W / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json').read_text())
idx = {f: i for i, f in enumerate(ids)}
out['real_vs_real_halves'] = lean([real[idx[f]] for f in split['selection_ids']], [real[idx[f]] for f in split['confirmation_ids']])
print('real_vs_real_halves', out['real_vs_real_halves'], flush=True)
Path(__file__).with_name('realism_select_v3.json').write_text(json.dumps(out, indent=1))
