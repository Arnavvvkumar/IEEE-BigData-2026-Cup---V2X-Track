"""Zip a candidate folder for upload (files at the zip root: synthetic/*.bin, predictions.json, and declaration.txt/.pdf if
present in the folder). Re-reads the zip and checks it the way the scorer does. usage: make_zip.py <candidate dir> <out.zip>"""
import os
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
src, out = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted((src / 'synthetic').glob('*.bin'))
assert len(files) == 50
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in files:
        z.write(f, f'synthetic/{f.name}')
    z.write(src / 'predictions.json', 'predictions.json')
    for d in ('declaration.txt', 'declaration.pdf'):
        if (src / d).exists():
            z.write(src / d, d)
with zipfile.ZipFile(out) as z:
    names = z.namelist()
    total = sum(i.file_size for i in z.infolist())
    bins = [n for n in names if n.startswith('synthetic/') and n.endswith('.bin')]
    assert len(bins) == 50 and all(z.getinfo(n).file_size <= 1_500_000 and z.getinfo(n).file_size % 12 == 0 for n in bins)
    pred = json.loads(z.read('predictions.json'))
    ids = json.loads((W / 'starting_kit/track_b/detection_test_frame_ids.json').read_text())
    assert set(pred) == set(ids)
    assert all(len(d['box']) == 7 and 'score' in d and d['class'] in ('vehicle', 'pedestrian', 'truck') for v in pred.values() for d in v)
    pts = np.frombuffer(z.read(bins[0]), dtype='<f4').reshape(-1, 3)
    decl = [n for n in names if n.startswith('declaration')]
print('zip', out, 'bytes', out.stat().st_size, '| uncompressed MB', round(total / 1e6, 1), '| entries', len(names), '| top-level', sorted({n.split('/')[0] for n in names}))
print('first scan points', len(pts), '| detections', sum(len(v) for v in pred.values()), '| declaration inside:', decl or 'NONE - must be added before upload')
