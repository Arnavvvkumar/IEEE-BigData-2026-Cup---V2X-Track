"""Pre-upload audit of a submission zip against the v9 zip. usage: verify_zip_any.py <zip name in Downloads> [reference zip]"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

DL = Path('/mnt/c/Users/arnav/Downloads')
W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
name = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else 'bdc2026_trackB_submission_v9.zip'
z = zipfile.ZipFile(DL / name)
names = z.namelist()
print(name, '| entries', len(names), '| top-level', sorted({n.split('/')[0] for n in names}))
ids = json.loads((W / 'starting_kit/track_b/detection_test_frame_ids.json').read_text())
p = json.loads(z.read('predictions.json'))
print('frame keys equal the 50 test ids:', set(p) == set(ids))
allb = [d for v in p.values() for d in v]
sc = np.array([d['score'] for d in allb]); bx = np.array([d['box'] for d in allb], float)
print('detections', len(allb), '| score min/max', float(sc.min()), round(float(sc.max()), 4), '| all in [0,1]:', bool(((sc >= 0) & (sc <= 1)).all()))
print('boxes finite:', bool(np.isfinite(bx).all()), '| 7 numbers each:', all(len(d['box']) == 7 for d in allb), '| sizes > 0:', bool((bx[:, 3:6] > 0).all()),
      '| classes:', sorted({d['class'] for d in allb}))
print('uncompressed MB', round(sum(i.file_size for i in z.infolist()) / 1e6, 1), '(limit about 75)')
print('declaration:', [l for l in z.read('declaration.txt').decode().splitlines() if l.strip()][-3:])
syn = sorted(n for n in names if n.startswith('synthetic/') and n.endswith('.bin'))
import os
SYN = os.environ.get('SYN', 'final_synthetic_v3')
v3 = sorted((W / f'experiments/claude_carla_overnight/twin/{SYN}/synthetic').glob('*.bin'))
print('synthetic files', len(syn), '| each <= 1.5 MB:', all(z.getinfo(n).file_size <= 1_500_000 for n in syn),
      '| identical to final_synthetic_v3:', all(hashlib.md5(z.read(n)).hexdigest() == hashlib.md5(f.read_bytes()).hexdigest() for n, f in zip(syn, v3)))
r = json.loads(zipfile.ZipFile(DL / ref).read('predictions.json'))
for c in ('vehicle', 'pedestrian', 'truck'):
    a = [d for v in r.values() for d in v if d['class'] == c]; b = [d for v in p.values() for d in v if d['class'] == c]
    print(f'{c}: {ref.split("_")[-1][:-4]} {len(a)} -> {len(b)} boxes')
tr = lambda q: [(tuple(d['box']), round(d['score'], 9)) for f in sorted(q) for d in q[f] if d['class'] == 'truck']
print('truck output identical to the reference zip:', tr(r) == tr(p))
