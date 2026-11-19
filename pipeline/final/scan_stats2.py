"""Are the released test scans the dev-style scans cropped to the kit range? Plane comparison (points only)."""
import os
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/erroranalysis'))
from groundplane import fit_ground  # noqa: E402

lo, hi = np.array([-40, -40, -8.0]), np.array([40, 40, 2.0])
for name, d in (('dev', W / 'experiments/real_reference_audit/real_dev_cw1/detection_test_frames'),
                ('test', W / 'starting_kit/track_b/detection_test_frames')):
    rows, planes = [], []
    for i, f in enumerate(sorted(d.glob('*.bin'))):
        p = np.fromfile(f, dtype='<f4').reshape(-1, 3)
        p = p[np.isfinite(p).all(1)]
        nz = p[np.abs(p).sum(1) > 0]
        inm = ((nz >= lo) & (nz <= hi)).all(1)
        rows.append((len(nz), int(inm.sum()), len(p) - len(nz), *np.abs(nz[:, :2]).max(0), nz[:, 2].min(), nz[:, 2].max()))
        planes.append(fit_ground(p, np.random.default_rng(i)))
    a = np.array(rows, float)
    print(name, 'nonzero med', np.median(a[:, 0]), 'in-mask med', np.median(a[:, 1]), 'in-mask+zeros med', np.median(a[:, 1] + a[:, 2]),
          '| max|x|', a[:, 3].max().round(1), 'max|y|', a[:, 4].max().round(1), 'z', a[:, 5].min().round(1), a[:, 6].max().round(1))
    P = np.array(planes)
    print(name, 'plane median', np.median(P, 0).round(4).tolist(), 'std', P.std(0).round(4).tolist())
