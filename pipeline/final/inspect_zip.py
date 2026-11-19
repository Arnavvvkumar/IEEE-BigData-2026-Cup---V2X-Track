"""READ-ONLY inspection of a submission zip the way the scorer would see it. usage: inspect_zip.py <zip>"""
import os
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
print('zip bytes', Path(sys.argv[1]).stat().st_size, '| entries', len(names), '| testzip (CRC):', z.testzip())
print('top-level:', sorted({n.split('/')[0] for n in names}))
bins = sorted(n for n in names if n.startswith('synthetic/') and n.endswith('.bin'))
sizes = [z.getinfo(n).file_size for n in bins]
print('scans', len(bins), '| names', bins[0], '...', bins[-1], '| max bytes', max(sizes), '(cap 1,500,000) | all multiple of 12:', all(s % 12 == 0 for s in sizes))
arr = [np.frombuffer(z.read(n), dtype='<f4').reshape(-1, 3) for n in bins]
print('finite', all(np.isfinite(a).all() for a in arr), '| points median', int(np.median([len(a) for a in arr])))
v2 = zipfile.ZipFile(os.environ['BDC_OUT'] + '/bdc2026_trackB_submission_v2.zip')
same = all(v2.read(n) == z.read(n) for n in bins)
print('scans identical to the submitted v2 scans (so realism unchanged):', same)
pred = json.loads(z.read('predictions.json'))
ids = json.loads((W / 'starting_kit/track_b/detection_test_frame_ids.json').read_text())
print('frame ids match kit exactly:', set(pred) == set(ids), len(pred))
C = ('vehicle', 'pedestrian', 'truck')
bad = 0
cnt = {c: 0 for c in C}
for f, v in pred.items():
    for d in v:
        ok = (set(d) == {'box', 'score', 'class'} and d['class'] in C and len(d['box']) == 7 and np.isfinite(d['box']).all()
              and 0 <= d['score'] <= 1 and min(d['box'][3:6]) > 0)
        bad += not ok
        cnt[d['class']] += 1
print('detections', cnt, '| malformed', bad)
decl = z.read('declaration.txt').decode('utf-8', errors='replace')
print('--- declaration.txt ---')
print(decl)
print('scorer phrase present:', 'I confirm that I did not train on any frame listed in'.lower() in decl.lower())
print('uncompressed MB', round(sum(i.file_size for i in z.infolist()) / 1e6, 1), '(limit ~75)')
