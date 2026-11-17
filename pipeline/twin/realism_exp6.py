"""Realism experiment F: histogram-matching thinning with the no-return share held FIXED (the previous run let the zero share drift).
Per scan: thin the non-zero points by the cross-fitted (range / range x height) keep probabilities, then set the zero count so zeros are exactly
a fraction `share` of that scan's points. share 0.296 = what the current recipe (zeros x0.85) gives; 0.308 = the real dev share. Kit realism, 5 seeds,
scored on the half the ratios were NOT fitted on. Dev frames only. Writes nothing to the submission."""
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


def keep_prob(ref, mode):
    r, s = nz(ref), nz(syn)
    cr, n = cells(r, mode)
    cs, _ = cells(s, mode)
    hr, hs, cnt = np.bincount(cr, minlength=n) / len(r), np.bincount(cs, minlength=n) / len(s), np.bincount(cs, minlength=n)
    ratio = np.where(cnt >= 2000, hr / np.maximum(hs, 1e-12), np.nan)
    return np.where(np.isnan(ratio), 1.0, np.clip(ratio / np.nanmax(ratio), 0.3, 1.0))


BOX = (-40, -40, -8, 40, 40, 2)


def inbox(q):
    return (q[:, 0] >= BOX[0]) & (q[:, 0] <= BOX[3]) & (q[:, 1] >= BOX[1]) & (q[:, 1] <= BOX[4]) & (q[:, 2] >= BOX[2]) & (q[:, 2] <= BOX[5])


def make(mode, kp, share, seed=0):
    """share = zero fraction among the points INSIDE the scoring box (what the realism metric sees)."""
    rng = np.random.default_rng(seed)
    out = []
    for p in syn:
        q = p[np.abs(p).sum(1) > 0]
        if kp is not None:
            q = q[rng.random(len(q)) < kp[cells(q, mode)[0]]]
        n_in = int(inbox(q).sum())
        out.append(np.vstack([q, np.zeros((int(round(share / (1 - share) * n_in)), 3))]))
    return out


zin = [(np.abs(p).sum(1) == 0).sum() / max(1, int(inbox(p).sum())) for p in syn]
print('zero share INSIDE the scoring box of the submitted v1 scans:', round(float(np.mean(zin)), 3), '(the realism metric sees this one)', flush=True)
s0 = make(None, None, 0.296)
print('check: control zero share inside the box:', round(float(np.mean([(np.abs(p).sum(1) == 0).sum() / int(inbox(p).sum()) for p in s0])), 3), flush=True)
for label, mode, share in (('control: no thinning, share 0.296', None, 0.296), ('range thin, share 0.296', 'range', 0.296), ('range thin, share 0.308', 'range', 0.308),
                           ('range x height thin, share 0.296', 'rz', 0.296), ('range x height thin, share 0.308', 'rz', 0.308)):
    res, em, pts = {}, {}, []
    for h, other in (('SEL', 'CONF'), ('CONF', 'SEL')):
        s = make(mode, keep_prob(HALF[other], mode) if mode else None, share)
        pts.append(np.mean([len(x) for x in s]))
        r = [T.lean(s, HALF[h], seed=sd) for sd in SEEDS]
        res[h] = (np.mean([x['realism'] for x in r]), np.std([x['realism'] for x in r]))
        em[h] = np.mean([x['EMD_n'] for x in r])
    print(f'{label:36s} SEL {res["SEL"][0]:.4f} (sd {res["SEL"][1]:.4f}, EMD_n {em["SEL"]:.3f}) | CONF {res["CONF"][0]:.4f} (sd {res["CONF"][1]:.4f}, EMD_n {em["CONF"]:.3f}) | points/scan {np.mean(pts):.0f}', flush=True)
print('REALISM_EXP6_DONE')
