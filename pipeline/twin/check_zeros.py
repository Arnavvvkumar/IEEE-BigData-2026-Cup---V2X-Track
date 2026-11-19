"""How many real points sit at/near the sensor origin, and does the kit's range mask keep them? (points only)"""
import os
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'starting_kit/track_b/scoring_program'))
import metrics as M  # noqa: E402

KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
RANGE = (-40.0, -40.0, -8.0, 40.0, 40.0, 2.0)
rows = []
for f in ids[:50]:
    p = np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3)
    r = np.linalg.norm(p, axis=1)
    m = M._mask_to_range(p, RANGE)
    rm = np.linalg.norm(m, axis=1)
    rows.append((len(p), int((r == 0).sum()), int((r < 0.5).sum()), int((r < 2).sum()), len(m), int((rm < 0.5).sum())))
a = np.array(rows)
print('median per scan: total, exact-zero, r<0.5, r<2, in-range-mask, in-mask r<0.5')
print(np.median(a, 0).astype(int).tolist())
print('min/max exact-zero', a[:, 1].min(), a[:, 1].max())
kit_test = W / 'starting_kit/track_b'
print([str(x.relative_to(kit_test)) for x in kit_test.rglob('*.bin')][:3])
