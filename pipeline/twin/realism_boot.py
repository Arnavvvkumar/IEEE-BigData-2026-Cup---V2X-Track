"""How trustworthy is a local realism gain? Bootstrap the official realism metric over random 50-frame REAL reference sets drawn from the
unlabelled real dev pools (dev150 + dev350 scans, plus the cw1 dev frames) and random subsample seeds, comparing two synthetic sets on the SAME
draw. Reports mean difference, sd, and P(B > A). Calibration case: v4 vs v3 (official: v4 0.9449 < v3 0.9507).
The released test scans and the official reference are never read.
usage: realism_boot.py <synthetic dir A> <synthetic dir B> <draws> <workers> [pool: all|dev150|dev350|cw1]"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import range_bg as T  # noqa: E402

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
COOP = W / 'experiments/claude_carla_overnight/coop'
load = lambda d: [np.fromfile(f, dtype='<f4').reshape(-1, 3) for f in sorted(Path(d).glob('*.bin'))]
A, B = load(sys.argv[1]), load(sys.argv[2])
draws, workers = int(sys.argv[3]), int(sys.argv[4])
which = sys.argv[5] if len(sys.argv) > 5 else 'all'
pools = {'dev150': load(COOP / 'dev150/scans'), 'dev350': load(COOP / 'dev350/scans'), 'cw1': list(T.clouds.values())}
pool = sum((v for k, v in pools.items() if which in ('all', k)), [])
print('pool sizes', {k: len(v) for k, v in pools.items()}, '| using', which, len(pool),
      '| points/scan real median', int(np.median([len(p) for p in pool])), 'zeros', int(np.median([(np.abs(p).sum(1) == 0).sum() for p in pool])),
      '| A', len(A), int(np.median([len(p) for p in A])), '| B', len(B), int(np.median([len(p) for p in B])), flush=True)


def one(k):
    rng = np.random.default_rng(1000 + k)
    ref = [pool[i] for i in rng.choice(len(pool), 50, replace=False)]
    seed = 1234 if k == 0 else int(rng.integers(1 << 30))
    a, b = T.lean(A, ref, seed=seed), T.lean(B, ref, seed=seed)
    return k, a, b


if __name__ == '__main__':
    from multiprocessing import get_context
    t0 = time.time()
    da, db = [], []
    with get_context('fork').Pool(workers) as p:
        for k, a, b in p.imap_unordered(one, range(draws)):
            da.append(a['realism']); db.append(b['realism'])
            d = np.array(db) - np.array(da)
            print(f'[{(time.time() - t0) / 60:5.1f} min] draw {k:3d} A {a["realism"]:.4f} (EMD_n {a["EMD_n"]:.3f}) B {b["realism"]:.4f} '
                  f'(EMD_n {b["EMD_n"]:.3f}) | running: n {len(d)} mean B-A {d.mean():+.4f} sd {d.std():.4f} P(B>A) {np.mean(d > 0):.2f}', flush=True)
    d = np.array(db) - np.array(da)
    print(f'BOOT_DONE n {len(d)} | A mean {np.mean(da):.4f} sd {np.std(da):.4f} | B mean {np.mean(db):.4f} sd {np.std(db):.4f} | '
          f'B-A mean {d.mean():+.4f} sd {d.std():.4f} P(B>A) {np.mean(d > 0):.2f}')
