"""Two small realism experiments (dev frames only; the released test scans are never used):
 A) WHERE the synthetic points differ from real ones inside the scoring box: share of points by range band, height band and
    azimuth sector (zeros reported separately). Seconds.
 B) Is trimming the no-return zeros a real gain or subsample luck? Zero count scaled by k, kit realism averaged over 5 subsample
    seeds on BOTH dev halves (the single-seed numbers jumped around: x0.90 +0.004 but x0.95 -0.002).
usage: realism_exp2.py"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import range_bg as T  # noqa: E402

syn = [np.fromfile(f, dtype='<f4').reshape(-1, 3).astype(np.float64) for f in sorted((HERE / 'final_synthetic_v1/synthetic').glob('*.bin'))]
SEL = [T.clouds[f] for f in T.split['selection_ids']]
CONF = [T.clouds[f] for f in T.split['confirmation_ids']]
B = (-40, -40, -8, 40, 40, 2)


def pool(clouds):
    p = np.concatenate([np.asarray(c, np.float64) for c in clouds])
    return p[(p[:, 0] >= B[0]) & (p[:, 0] <= B[3]) & (p[:, 1] >= B[1]) & (p[:, 1] <= B[4]) & (p[:, 2] >= B[2]) & (p[:, 2] <= B[5])]


print('=== A) share of in-box points (synthetic vs real dev, all 50+ frames each)', flush=True)
S, R = pool(syn), pool(SEL + CONF)
for lab, a in (('synthetic', S), ('real dev', R)):
    z = np.abs(a).sum(1) == 0
    print(f'  {lab:10s} zeros {100 * z.mean():5.1f}%', flush=True)
S, R = S[np.abs(S).sum(1) > 0], R[np.abs(R).sum(1) > 0]
rows = [('range m', lambda p: np.hypot(p[:, 0], p[:, 1]), [0, 5, 10, 15, 20, 25, 30, 40, 57]),
        ('height z m', lambda p: p[:, 2], [-8, -3.5, -3, -2.6, -2, -1, 0, 1, 2]),
        ('azimuth deg', lambda p: np.degrees(np.arctan2(p[:, 1], p[:, 0])), list(range(-180, 181, 45)))]
for lab, fn, bins in rows:
    hs = np.histogram(fn(S), bins)[0] / len(S); hr = np.histogram(fn(R), bins)[0] / len(R)
    print(f'  --- non-zero points by {lab}: synthetic% / real% (diff)', flush=True)
    for i in range(len(bins) - 1):
        print(f'     {bins[i]:6.1f}..{bins[i + 1]:6.1f}: {100 * hs[i]:5.1f} / {100 * hr[i]:5.1f}  ({100 * (hs[i] - hr[i]):+5.1f})', flush=True)

print('=== B) zero trimming, kit realism averaged over 5 seeds, both halves', flush=True)
SEEDS = (1234, 1, 2, 3, 4)
for k in (1.0, 0.95, 0.93, 0.90, 0.85):
    s = []
    for p in syn:
        z = np.abs(p).sum(1) == 0
        s.append(np.vstack([p[~z], np.zeros((int(round(z.sum() * k)), 3))]))
    r = {h: [T.lean(s, ref, seed=sd) for sd in SEEDS] for h, ref in (('SEL', SEL), ('CONF', CONF))}
    m = {h: np.mean([x['realism'] for x in v]) for h, v in r.items()}
    e = {h: np.mean([x['EMD_n'] for x in v]) for h, v in r.items()}
    sd = {h: np.std([x['realism'] for x in v]) for h, v in r.items()}
    print(f'  zeros x{k:4.2f}: SEL {m["SEL"]:.4f} (sd {sd["SEL"]:.4f}, EMD_n {e["SEL"]:.3f}) | CONF {m["CONF"]:.4f} (sd {sd["CONF"]:.4f}, EMD_n {e["CONF"]:.3f})', flush=True)
print('REALISM_EXP2_DONE')
