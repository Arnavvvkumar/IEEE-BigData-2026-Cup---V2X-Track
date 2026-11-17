"""Final pre-upload audit of bdc2026_trackB_submission_v9.zip: format, scores, boxes, declaration, synthetic scans, and content vs v8."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

DL = Path('/mnt/c/Users/arnav/Downloads')
W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
z = zipfile.ZipFile(DL / 'bdc2026_trackB_submission_v9.zip')
names = z.namelist()
print('entries', len(names), '| top-level', sorted({n.split('/')[0] for n in names}))
ids = json.loads((W / 'starting_kit/track_b/detection_test_frame_ids.json').read_text())
p = json.loads(z.read('predictions.json'))
print('frame keys equal the 50 test ids:', set(p) == set(ids), len(p))
allb = [d for v in p.values() for d in v]
sc = np.array([d['score'] for d in allb]); bx = np.array([d['box'] for d in allb], float)
print('detections', len(allb), '| score min/max', float(sc.min()), float(sc.max()), '| all in [0,1]:', bool(((sc >= 0) & (sc <= 1)).all()))
print('boxes finite:', bool(np.isfinite(bx).all()), '| 7 numbers each:', all(len(d['box']) == 7 for d in allb), '| sizes > 0:', bool((bx[:, 3:6] > 0).all()))
print('class counts', {c: sum(d['class'] == c for d in allb) for c in ('vehicle', 'pedestrian', 'truck')}, '| classes present only these three:', {d['class'] for d in allb} <= {'vehicle', 'pedestrian', 'truck'})
print('uncompressed MB', round(sum(i.file_size for i in z.infolist()) / 1e6, 1), '(limit about 75)')
decl = z.read('declaration.txt').decode()
print('declaration:'); print('   ' + decl.replace('\n', '\n   '))
syn = sorted(n for n in names if n.startswith('synthetic/') and n.endswith('.bin'))
v3 = sorted((W / 'experiments/claude_carla_overnight/twin/final_synthetic_v3/synthetic').glob('*.bin'))
same = all(hashlib.md5(z.read(n)).hexdigest() == hashlib.md5(f.read_bytes()).hexdigest() for n, f in zip(syn, v3))
print('synthetic files', len(syn), '| each <= 1.5 MB:', all(z.getinfo(n).file_size <= 1_500_000 for n in syn), '| identical to final_synthetic_v3 (zeros x0.85):', same)
a = np.frombuffer(z.read(syn[0]), '<f4').reshape(-1, 3)
print('first scan points', len(a), '| zeros share', round(float((np.abs(a).sum(1) == 0).mean()), 3))
v8 = json.loads(zipfile.ZipFile(DL / 'bdc2026_trackB_submission_v8.zip').read('predictions.json'))
hits = lambda q, c: {f: sum(d['class'] == c for d in q[f]) for f in q}
for c in ('vehicle', 'pedestrian', 'truck'):
    print(f'{c}: v8 {sum(hits(v8, c).values())} -> v9 {sum(hits(p, c).values())} boxes')
