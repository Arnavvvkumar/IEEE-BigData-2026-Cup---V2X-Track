"""Site twin, step 2 (CPU): range-image background model of the fixed real sensor, from UNLABELLED real clouds only.
For every (elevation 0.25 deg, azimuth 1024/rev) cell: the background range = the far, repeatable return across scans
(moving objects give shorter, non-repeating ranges and are rejected), its spread, its mean direction and its return probability.
Sampling the model gives a synthetic background scan with the real site's layout and the real sensor's beam pattern.
Honest check: build on the selection half, score realism against the confirmation half (and the reverse).
Real-data use to disclose: unlabelled real training clouds (non-forbidden) for site/sensor calibration."""
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
HERE = Path(__file__).parent
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/realism'))
KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
split = json.loads((W / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json').read_text())
clouds = {f: np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
E0, E1, DE, NA = -17.5, 17.0, 0.25, 1024
NE = int(round((E1 - E0) / DE))


def cells(p):
    r = np.linalg.norm(p, axis=1)
    ok = r > 0.5
    p, r = p[ok], r[ok]
    el = np.degrees(np.arcsin(p[:, 2] / r))
    az = np.arctan2(p[:, 1], p[:, 0])
    ie = np.floor((el - E0) / DE).astype(int)
    ia = np.floor((az + np.pi) / (2 * np.pi) * NA).astype(int) % NA
    ok = (ie >= 0) & (ie < NE)
    return ie[ok] * NA + ia[ok], r[ok], p[ok] / r[ok, None]


def build(frames, tol=0.03, min_frac=0.2):
    """returns dict of arrays over cells that have a repeatable background return."""
    cid, rr, dd, fid = [], [], [], []
    for i, f in enumerate(frames):
        c, r, d = cells(clouds[f])
        cid.append(c); rr.append(r); dd.append(d); fid.append(np.full(len(c), i))
    cid, rr, dd, fid = np.concatenate(cid), np.concatenate(rr), np.concatenate(dd), np.concatenate(fid)
    order = np.lexsort((rr, cid))
    cid, rr, dd, fid = cid[order], rr[order], dd[order], fid[order]
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    ends = np.r_[starts[1:], len(cid)]
    n = len(frames)
    out = {k: [] for k in ('cell', 'range', 'sigma', 'prob', 'dir')}
    for s, e in zip(starts, ends):
        r = rr[s:e]
        if e - s < min_frac * n:
            continue
        ref = r[int(0.8 * (len(r) - 1))]  # far, repeatable surface
        m = np.abs(r - ref) <= max(0.15, tol * ref)
        k = len(np.unique(fid[s:e][m]))
        if k < min_frac * n:
            continue
        d = dd[s:e][m].mean(0)
        out['cell'].append(cid[s]); out['range'].append(np.median(r[m])); out['sigma'].append(r[m].std())
        out['prob'].append(min(1.0, m.sum() / n)); out['dir'].append(d / np.linalg.norm(d))
    return {k: np.array(v) for k, v in out.items()}


def sample(model, rng, ang_jitter=0.0015):
    keep = rng.random(len(model['prob'])) < model['prob']
    d = model['dir'][keep] + rng.normal(0, ang_jitter, (keep.sum(), 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = model['range'][keep] + rng.normal(0, 1, keep.sum()) * np.clip(model['sigma'][keep], 0.01, 0.2)
    return (d * r[:, None]).astype(np.float32)


def lean(syn, ref, seed=1234):
    """kit realism metric with chunked MMD (same maths and RNG order as realism_select_v3.lean)."""
    from scipy.spatial.distance import cdist
    sys.path.insert(0, str(W / 'starting_kit/track_b/scoring_program'))
    import metrics as M
    RANGE = (-40.0, -40.0, -8.0, 40.0, 40.0, 2.0)
    NORM = {'CD': (2.0, 30.0), 'MMD': (0.0, 0.5), 'EMD': (0.0, 5.0), 'FPD': (0.0, 50.0)}
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
    return {'realism': round(float(np.mean(list(s.values()))), 4), **{k + '_n': round(v, 3) for k, v in s.items()}}


if __name__ == '__main__':
    res = {}
    sel, conf = split['selection_ids'], split['confirmation_ids']
    for name, a, b in (('build_sel_score_conf', sel, conf), ('build_conf_score_sel', conf, sel)):
        model = build(a)
        rng = np.random.default_rng(7)
        scans = [sample(model, rng) for _ in range(50)]
        res[name] = {**lean(scans, [clouds[f] for f in b]), 'cells': int(len(model['cell'])),
                     'points_per_scan': int(np.median([len(s) for s in scans]))}
        print(name, res[name], flush=True)
    res['real_vs_real'] = lean([clouds[f] for f in sel], [clouds[f] for f in conf])
    print('real_vs_real', res['real_vs_real'], flush=True)
    model = build(ids)
    np.savez_compressed(HERE / 'range_bg_all50.npz', **model)
    print('all-50 model cells', len(model['cell']), 'real points/scan median', int(np.median([len(clouds[f]) for f in ids])))
    (HERE / 'range_bg_v1.json').write_text(json.dumps(res, indent=1))
