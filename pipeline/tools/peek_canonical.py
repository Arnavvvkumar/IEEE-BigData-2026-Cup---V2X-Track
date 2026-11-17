"""Read-only peek at the canonical Generator V2 dataset and the OpenPCDet copy (format + box/ground stats)."""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path.home() / 'projects/bdc2026-track-b/data/claude_v2_scale5500_fix1'
PCD = Path.home() / 'projects/bdc2026-track-b/data/codex_openpcdet_custom_v1'
recs = json.loads((SRC / 'frames.json').read_text())
r0 = recs[0]
print('record keys:', sorted(r0))
print('example:', {k: v for k, v in r0.items() if k != 'ground_truth'})
print('gt0:', r0['ground_truth'][:2])
print('splits:', {s: sum(r['split'] == s for r in recs) for s in {r['split'] for r in recs}})
print('top dirs:', sorted(p.name for p in SRC.iterdir())[:10])
print('pcd dirs:', sorted(p.name for p in PCD.iterdir()))
cfg = Path.home() / 'projects/bdc2026-track-b/data/claude_v2_scale5500_fix1/config.json'
if cfg.exists():
    print('config:', cfg.read_text()[:800])
rng = np.random.default_rng(0)
train = [r for r in recs if r['split'] == 'train']
by = {}
for r in train:
    for b in r['ground_truth']:
        by.setdefault(b['class'], []).append(b['box'])
for c, v in by.items():
    a = np.asarray(v)
    print(c, len(a), 'median lwh', np.round(np.median(a[:, 3:6], 0), 2), 'p10/p90 l', np.round(np.percentile(a[:, 3], [10, 90]), 2),
          'median bottom z', round(float(np.median(a[:, 2] - a[:, 5] / 2)), 2), 'median range', round(float(np.median(np.hypot(a[:, 0], a[:, 1]))), 1))
pts_files = sorted((PCD / 'points').glob('*.npy'))
sample = [pts_files[i] for i in rng.choice(len(pts_files), 20, replace=False)]
inten, n, zmin = [], [], []
for f in sample:
    p = np.load(f)
    inten.append(np.percentile(p[:, 3], [5, 50, 95]))
    n.append(len(p))
    zmin.append(np.percentile(p[:, 2], 2))
    xyext = (p[:, :2].min(0), p[:, :2].max(0), p[:, 2].min(), p[:, 2].max())
print('npy dtype', np.load(sample[0]).dtype, 'shape', np.load(sample[0]).shape)
print('points median', np.median(n), 'intensity p5/50/95 median', np.round(np.median(inten, 0), 3), 'z p2 median', np.round(np.median(zmin), 2))
print('last extent', xyext)
print(open(PCD / 'labels' / (sample[0].stem + '.txt')).read()[:300])
sys.stdout.flush()
