"""Packaging equality only; no test-data diagnostics or candidate selection from content."""
import os
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(os.environ['BDC_OUT'])
versions = ['v15s', 'v17s_d15', 'v18s']
pred, scans = {}, {}
for version in versions:
    path = root / f'bdc2026_trackB_submission_{version}.zip'
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        pred[version] = json.loads(z.read('predictions.json'))
        scans[version] = {n: hashlib.sha256(z.read(n)).hexdigest()
                          for n in z.namelist() if n.startswith('synthetic/') and n.endswith('.bin')}
    print(version, 'sha256', hashlib.sha256(path.read_bytes()).hexdigest())
def cls(p, c):
    return {f: [d for d in ds if d['class'] == c] for f, ds in p.items()}
for a, b in [('v15s', 'v17s_d15'), ('v17s_d15', 'v18s')]:
    print(a, b, 'same_frame_ids', set(pred[a]) == set(pred[b]),
          'same_synthetic_files', scans[a] == scans[b],
          'class_exact_equal', {c: cls(pred[a], c) == cls(pred[b], c)
                                for c in ['vehicle', 'pedestrian', 'truck']})
