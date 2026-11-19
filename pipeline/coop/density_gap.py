"""Is the point density of our SYNTHETIC training pedestrians/vehicles different from REAL ones at the same range?
Synthetic: claude_canon_all_twin_st_v1 train frames (points/*.npy, labels/*.txt), points inside each box (all points).
Real: dev150 scans + gt_coop.json boxes (>= 5 sensor points), points inside each box (all points, no zeros).
Prints median / quartile in-box point counts per range band, and the ratio synthetic / real.
usage: density_gap.py [n_synthetic_frames=1500]"""
import os
import glob
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
D = Path.home() / 'projects/bdc2026-track-b/data/claude_canon_all_twin_st_v1'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
BANDS = [(0, 10), (10, 15), (15, 20), (20, 30), (30, 45)]


def count(p, b):
    r = p - b[:3]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    x = r[:, 0] * cs - r[:, 1] * sn; y = r[:, 0] * sn + r[:, 1] * cs
    return int(((np.abs(x) <= b[3] / 2) & (np.abs(y) <= b[4] / 2) & (np.abs(r[:, 2]) <= b[5] / 2)).sum())


print('dataset', D, 'exists', D.exists(), flush=True)
lab = sorted(glob.glob(str(D / 'labels/*.txt')))
rng = np.random.default_rng(0)
pick = rng.choice(len(lab), min(N, len(lab)), replace=False)
syn = {'pedestrian': [], 'vehicle': [], 'truck': []}
for i in pick:
    f = Path(lab[i])
    rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
    pts = np.load(D / 'points' / (f.stem + '.npy'))[:, :3].astype(float)
    for r in rows:
        cls = r[7].lower()
        if cls not in syn:
            continue
        b = np.array(r[:7], float)
        rr = float(np.hypot(b[0], b[1]))
        near = pts[np.hypot(pts[:, 0] - b[0], pts[:, 1] - b[1]) < max(b[3], b[4]) + 1.0]
        syn[cls].append((rr, count(near, b)))
real = {'pedestrian': [], 'vehicle': [], 'truck': []}
d = W / 'experiments/claude_carla_overnight/coop/dev150'
G = json.loads((d / 'gt_coop.json').read_text())
for fid, v in G.items():
    p = np.fromfile(d / 'scans' / f'{fid}.bin', dtype='<f4').reshape(-1, 3).astype(float)
    p = p[np.abs(p).sum(1) > 0]
    for g in v:
        if g['points'] >= 5 and g['class'] in real:
            b = np.array(g['box'], float)
            near = p[np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) < max(b[3], b[4]) + 1.0]
            real[g['class']].append((float(np.hypot(b[0], b[1])), count(near, b)))
for cls in ('pedestrian', 'vehicle', 'truck'):
    print(f'== {cls}: synthetic n={len(syn[cls])}, real n={len(real[cls])}')
    for lo, hi in BANDS:
        s = np.array([c for r, c in syn[cls] if lo <= r < hi]); rl = np.array([c for r, c in real[cls] if lo <= r < hi])
        if len(s) < 20 or len(rl) < 20:
            print(f'  {lo:2d}-{hi:2d} m: synthetic n={len(s)}, real n={len(rl)} (too few)'); continue
        qs, qr = np.percentile(s, [25, 50, 75]), np.percentile(rl, [25, 50, 75])
        print(f'  {lo:2d}-{hi:2d} m: synthetic q25/50/75 {qs.round(0).tolist()} (n={len(s)}) | real {qr.round(0).tolist()} (n={len(rl)}) | median ratio syn/real {qs[1] / max(qr[1], 1):.2f}', flush=True)
print('DENSITY_GAP_DONE')
