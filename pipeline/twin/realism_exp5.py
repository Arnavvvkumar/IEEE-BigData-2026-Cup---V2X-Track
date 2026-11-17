"""Realism experiment E: histogram matching by thinning. The synthetic scans have too many non-zero points at 5-10 m and too few at 20-40 m
(and some height-band mismatch). Thin non-zero points per (range x height) cell with keep probability proportional to
real_share / synthetic_share (cells with few points are left alone), combined with the no-return zero trim. Cross-fitted: the cell ratios are
estimated on one dev half and the scan is scored on the OTHER half. Kit realism averaged over 5 subsample seeds. Dev frames only; the released
test scans are never used. Writes nothing to the submission.
usage: realism_exp5.py"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import range_bg as T  # noqa: E402

syn = [np.fromfile(f, dtype='<f4').reshape(-1, 3).astype(np.float64) for f in sorted((HERE / 'final_synthetic_v1/synthetic').glob('*.bin'))]
HALF = {'SEL': [T.clouds[f] for f in T.split['selection_ids']], 'CONF': [T.clouds[f] for f in T.split['confirmation_ids']]}
SEEDS = (1234, 1, 2, 3, 4)
RB = np.array([0, 5, 10, 15, 20, 25, 30, 40, 57, 1e9])
ZB = np.array([-1e9, -3.5, -3.0, -2.6, -2.0, -1.0, 0.0, 1.0, 1e9])


def nz(clouds):
    p = np.concatenate([np.asarray(c, np.float64) for c in clouds])
    return p[np.abs(p).sum(1) > 0]


def cells(p, mode):
    ri = np.clip(np.digitize(np.hypot(p[:, 0], p[:, 1]), RB) - 1, 0, len(RB) - 2)
    if mode == 'range':
        return ri, len(RB) - 1
    zi = np.clip(np.digitize(p[:, 2], ZB) - 1, 0, len(ZB) - 2)
    return ri * (len(ZB) - 1) + zi, (len(RB) - 1) * (len(ZB) - 1)


def keep_prob(ref_clouds, mode):
    r, s = nz(ref_clouds), nz(syn)
    cr, n = cells(r, mode)
    cs, _ = cells(s, mode)
    hr = np.bincount(cr, minlength=n) / len(r)
    hs = np.bincount(cs, minlength=n) / len(s)
    cnt = np.bincount(cs, minlength=n)
    ratio = np.where(cnt >= 2000, hr / np.maximum(hs, 1e-12), np.nan)
    top = np.nanmax(ratio)
    k = np.where(np.isnan(ratio), 1.0, np.clip(ratio / top, 0.3, 1.0))
    return k


def apply(mode, kp, kz, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for p in syn:
        z = np.abs(p).sum(1) == 0
        q = p[~z]
        if kp is not None:
            c, _ = cells(q, mode)
            q = q[rng.random(len(q)) < kp[c]]
        out.append(np.vstack([q, np.zeros((int(round(z.sum() * kz)), 3))]))
    return out


for label, mode, kz in (('control: zeros x0.85 only', None, 0.85), ('range-only thin, zeros x1.00', 'range', 1.0), ('range-only thin, zeros x0.90', 'range', 0.9),
                        ('range-only thin, zeros x0.85', 'range', 0.85), ('range x height thin, zeros x1.00', 'rz', 1.0),
                        ('range x height thin, zeros x0.90', 'rz', 0.9), ('range x height thin, zeros x0.85', 'rz', 0.85)):
    res, eml, pts = {}, {}, []
    for h, other in (('SEL', 'CONF'), ('CONF', 'SEL')):       # ratios from `other`, score against `h`... cross-fit: fit on the other half
        kp = keep_prob(HALF[other], mode) if mode else None
        s = apply(mode, kp, kz)
        pts.append(np.mean([len(x) for x in s]))
        r = [T.lean(s, HALF[h], seed=sd) for sd in SEEDS]
        res[h] = (np.mean([x['realism'] for x in r]), np.std([x['realism'] for x in r]))
        eml[h] = np.mean([x['EMD_n'] for x in r])
    print(f'{label:36s} SEL {res["SEL"][0]:.4f} (sd {res["SEL"][1]:.4f}, EMD_n {eml["SEL"]:.3f}) | CONF {res["CONF"][0]:.4f} (sd {res["CONF"][1]:.4f}, EMD_n {eml["CONF"]:.3f}) | points/scan {np.mean(pts):.0f}', flush=True)
print('REALISM_EXP5_DONE')
