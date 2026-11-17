"""Realism experiment C: synthetic scans have too many in-box points at 5-10 m (34.9% vs 31.1% real) and too few at 20-40 m, plus
2.3 points too many zeros. Thin a random fraction of the non-zero 5-10 m (and optionally 10-15 m) points per scan, and optionally trim
zeros; score kit realism averaged over 5 subsample seeds on both dev halves. Dev frames only. Writes nothing to the submission.
usage: realism_exp3.py"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import range_bg as T  # noqa: E402

syn = [np.fromfile(f, dtype='<f4').reshape(-1, 3).astype(np.float64) for f in sorted((HERE / 'final_synthetic_v1/synthetic').glob('*.bin'))]
SEL = [T.clouds[f] for f in T.split['selection_ids']]
CONF = [T.clouds[f] for f in T.split['confirmation_ids']]
SEEDS = (1234, 1, 2, 3, 4)


def thin(p5, p10, kz, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for p in syn:
        z = np.abs(p).sum(1) == 0
        r = np.hypot(p[:, 0], p[:, 1])
        drop = np.zeros(len(p), bool)
        drop |= (~z) & (r >= 5) & (r < 10) & (rng.random(len(p)) < p5)
        drop |= (~z) & (r >= 10) & (r < 15) & (rng.random(len(p)) < p10)
        q = p[~drop & ~z]
        out.append(np.vstack([q, np.zeros((int(round(z.sum() * kz)), 3))]))
    return out


for p5, p10, kz in ((0, 0, 1), (0.10, 0, 1), (0.15, 0, 1), (0.20, 0, 1), (0.15, 0.05, 1), (0.15, 0, 0.93), (0.20, 0.05, 0.93)):
    s = thin(p5, p10, kz)
    r = {h: [T.lean(s, ref, seed=sd) for sd in SEEDS] for h, ref in (('SEL', SEL), ('CONF', CONF))}
    m = {h: np.mean([x['realism'] for x in v]) for h, v in r.items()}
    e = {h: np.mean([x['EMD_n'] for x in v]) for h, v in r.items()}
    print(f'thin 5-10m {p5:.2f}, 10-15m {p10:.2f}, zeros x{kz:.2f}: SEL {m["SEL"]:.4f} (EMD_n {e["SEL"]:.3f}) | CONF {m["CONF"]:.4f} (EMD_n {e["CONF"]:.3f})', flush=True)
print('REALISM_EXP3_DONE')
