"""Point statistics: released test scans vs dev frames (points only)."""
import json
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
for name, d in (('dev', W / 'experiments/real_reference_audit/real_dev_cw1/detection_test_frames'),
                ('test', W / 'starting_kit/track_b/detection_test_frames')):
    rows = []
    for f in sorted(d.glob('*.bin')):
        raw = np.fromfile(f, dtype='<f4')
        p = raw.reshape(-1, 3) if raw.size % 3 == 0 else raw.reshape(-1, 4)[:, :3]
        r = np.linalg.norm(p, axis=1)
        nz = p[r > 0]
        rows.append((len(p), int((r == 0).sum()), len(nz), float(np.median(r[r > 0])), float(np.percentile(r[r > 0], 95)),
                     int(len(np.unique(nz, axis=0))), raw.size % 3))
    a = np.array(rows, float)
    print(name, len(rows), 'files | n', np.median(a[:, 0]), a[:, 0].min(), a[:, 0].max(), '| zeros', np.median(a[:, 1]), a[:, 1].min(), a[:, 1].max(),
          '| nonzero', np.median(a[:, 2]), '| r med', a[:, 3].mean().round(2), 'r p95', a[:, 4].mean().round(2), '| unique nonzero', np.median(a[:, 5]),
          '| size%3', a[:, 6].max())
