"""Realism (kit metric, chunked MMD) of the first N twin scans (sensor frame) against the real dev frames, plus sanity stats.
usage: twin_realism.py <twin dataset dir> <N>"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import range_bg as T  # noqa: E402

d, n = Path(sys.argv[1]), int(sys.argv[2])
fs = sorted((d / 'sensor').glob('*.npy'))[:n]
scans = [np.load(f) for f in fs]
out = {'all50': T.lean(scans, [T.clouds[f] for f in T.ids]),
       'vs_selection': T.lean(scans, [T.clouds[f] for f in T.split['selection_ids']]),
       'vs_confirmation': T.lean(scans, [T.clouds[f] for f in T.split['confirmation_ids']]),
       'points_per_scan_median': int(np.median([len(s) for s in scans])), 'bytes_per_scan_max': int(max(s.nbytes for s in scans)),
       'manifest': json.loads((d / 'manifest.json').read_text())}
print(json.dumps(out, indent=1))
(Path(__file__).parent / f'twin_realism_{d.name}.json').write_text(json.dumps(out, indent=1))
