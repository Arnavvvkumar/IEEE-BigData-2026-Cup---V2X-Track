"""Realism experiment D: zero trimming kept improving down to x0.85 (5-seed mean, both dev halves: 0.9405 -> 0.9488 SEL,
0.9413 -> 0.9479 CONF). Find where it peaks: x0.80, 0.75, 0.70, 0.60. Dev frames only. Writes nothing.
usage: realism_exp4.py"""
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
for k in (0.80, 0.75, 0.70, 0.60):
    s = []
    for p in syn:
        z = np.abs(p).sum(1) == 0
        s.append(np.vstack([p[~z], np.zeros((int(round(z.sum() * k)), 3))]))
    r = {h: [T.lean(s, ref, seed=sd) for sd in SEEDS] for h, ref in (('SEL', SEL), ('CONF', CONF))}
    m = {h: np.mean([x['realism'] for x in v]) for h, v in r.items()}
    sd = {h: np.std([x['realism'] for x in v]) for h, v in r.items()}
    e = {h: np.mean([x['EMD_n'] for x in v]) for h, v in r.items()}
    print(f'zeros x{k:4.2f}: SEL {m["SEL"]:.4f} (sd {sd["SEL"]:.4f}, EMD_n {e["SEL"]:.3f}) | CONF {m["CONF"]:.4f} (sd {sd["CONF"]:.4f}, EMD_n {e["CONF"]:.3f}) | seed-1234 SEL {r["SEL"][0]["realism"]} CONF {r["CONF"][0]["realism"]}', flush=True)
print('REALISM_EXP4_DONE')
