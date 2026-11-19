"""Box-size statistics per class in the real dev labels (allowed use: box-size statistics)."""
import os
import json
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
gt = json.loads((W / 'experiments/real_reference_audit/real_dev_cw1/ground_truth.json').read_text())
by = {}
for v in gt.values():
    for g in v:
        by.setdefault(g['class'], []).append(g['box'][3:6])
for c, b in by.items():
    b = np.array(b)
    print(c, len(b), 'median l,w,h', np.median(b, 0).round(2).tolist(), 'length p5/p50/p95', np.percentile(b[:, 0], [5, 50, 95]).round(2).tolist(),
          'height p5/p95', np.percentile(b[:, 2], [5, 95]).round(2).tolist())
