"""final_synthetic_v3 = the submitted 50 synthetic scans (final_synthetic_v1) with the no-return zero points trimmed to 85% per scan
(simulated sensor dropout rate tuned on unlabelled real dev frames: 5-seed realism SEL 0.9405 -> 0.9488, CONF 0.9413 -> 0.9479).
Non-zero points are unchanged. Also reports the kit realism of the new set (seed 1234) on both dev halves.
usage: make_synthetic_v3.py"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import range_bg as T  # noqa: E402

K = 0.85
src, out = HERE / 'final_synthetic_v1/synthetic', HERE / 'final_synthetic_v3/synthetic'
out.mkdir(parents=True, exist_ok=True)
scans = []
for f in sorted(src.glob('*.bin')):
    p = np.fromfile(f, dtype='<f4').reshape(-1, 3)
    z = np.abs(p).sum(1) == 0
    nz = int(round(z.sum() * K))
    q = np.vstack([p[~z], np.zeros((nz, 3), np.float32)]).astype('<f4')
    q.tofile(out / f.name)
    scans.append(q.astype(np.float64))
back = [np.fromfile(f, dtype='<f4').reshape(-1, 3) for f in sorted(out.glob('*.bin'))]
assert len(back) == 50 and max(f.stat().st_size for f in out.glob('*.bin')) <= 1_500_000
print('wrote', len(back), 'scans | max bytes', max(f.stat().st_size for f in out.glob('*.bin')), '| median points', int(np.median([len(b) for b in back])))
for h, ids in (('SEL', T.split['selection_ids']), ('CONF', T.split['confirmation_ids'])):
    print(h, 'seed 1234:', T.lean(scans, [T.clouds[f] for f in ids]))
