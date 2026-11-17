"""Realism: greedy scan swaps from the 400-scan twin pool, starting from the submitted 50 (twin/final_synthetic_v1).
Objective = mean kit realism (lean, same maths as the scorer) over the dev SELECTION half with 3 subsample seeds (the single-seed
metric is noisy, about +-0.002). A swap is kept only if the 3-seed mean improves by >= 0.0005. Every accepted swap is also scored on
the dev CONFIRMATION half (seed 1234, the server's seed) and logged; the final set is adopted only if it beats the start on CONF.
Dev frames only (non-forbidden real dev cw1 split); the released test scans are never used. Output: twin/final_synthetic_v2/.
usage: realism_swap.py <minutes budget>"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import range_bg as T  # noqa: E402

budget = float(sys.argv[1]) * 60
POOL = sorted((Path.home() / 'projects/bdc2026-track-b/data/claude_twin_v1/sensor').glob('*.npy'))
log0 = json.loads((HERE / 'final_synthetic_v1/selection_log.json').read_text())
names = [p.name for p in POOL]
cur = [names.index(n) for n in log0['picked_files']]
cache = {}
get = lambda i: cache.setdefault(i, np.load(POOL[i])[:, :3].astype(np.float64))
SEL = [T.clouds[f] for f in T.split['selection_ids']]
CONF = [T.clouds[f] for f in T.split['confirmation_ids']]
SEEDS = (1234, 1, 2)


def obj(idx):
    s = [get(i) for i in idx]
    return float(np.mean([T.lean(s, SEL, seed=k)['realism'] for k in SEEDS]))


conf = lambda idx: T.lean([get(i) for i in idx], CONF, seed=1234)
t0 = time.time()
best = obj(cur)
c0 = conf(cur)
print(f'start: SEL 3-seed {best:.4f} | CONF {c0}', flush=True)
rng = np.random.default_rng(11)
tried = acc = 0
history = []
while time.time() - t0 < budget:
    out_i = int(rng.integers(50))
    cand = [i for i in range(len(POOL)) if i not in cur]
    new = list(cur); new[out_i] = int(rng.choice(cand))
    v = obj(new); tried += 1
    if v >= best + 0.0005:
        cur, best = new, v; acc += 1
        c = conf(cur)
        history.append({'tried': tried, 'sel3': round(best, 4), 'conf': c})
        print(f'[{(time.time() - t0) / 60:5.1f} min] accepted #{acc} after {tried} tries: SEL 3-seed {best:.4f} | CONF {c["realism"]} EMD_n {c["EMD_n"]}', flush=True)
cf = conf(cur)
print(f'end: tried {tried}, accepted {acc} | SEL 3-seed {best:.4f} | CONF {cf} (start CONF {c0["realism"]})', flush=True)
out = HERE / 'final_synthetic_v2'
if cf['realism'] > c0['realism']:
    (out / 'synthetic').mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(cur, 1):
        get(i).astype('<f4').tofile(out / 'synthetic' / f'{j:04d}.bin')
    (out / 'selection_log.json').write_text(json.dumps({'start_conf': c0, 'end_conf': cf, 'sel3_end': best, 'tried': tried, 'accepted': acc,
                                                       'picked_files': [names[i] for i in cur], 'history': history}, indent=1))
    print('ADOPTED: wrote', out, flush=True)
else:
    print('NOT ADOPTED: confirmation half did not improve', flush=True)
print('REALISM_SWAP_DONE')
