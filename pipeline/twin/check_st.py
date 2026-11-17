import numpy as np
from pathlib import Path
D = Path.home() / 'projects/bdc2026-track-b/data/claude_twin_st_v1'
sz, npts = [], []
for p in sorted((D / 'labels').glob('*.txt'))[:300]:
    pts = None
    for ln in p.read_text().splitlines():
        *n, c = ln.split()
        if c != 'truck':
            continue
        b = np.array(n, float)
        if pts is None:
            pts = np.load(D / 'points' / (p.stem + '.npy'))[:, :3]
        q = pts - b[:3]
        cs, sn = np.cos(-b[6]), np.sin(-b[6])
        x, y = q[:, 0] * cs - q[:, 1] * sn, q[:, 0] * sn + q[:, 1] * cs
        m = (abs(x) <= b[3] / 2 + .05) & (abs(y) <= b[4] / 2 + .05) & (abs(q[:, 2]) <= b[5] / 2 + .05)
        sz.append(b[3:6]); npts.append(m.sum())
sz = np.array(sz)
print('trucks', len(sz), 'L/W/H median', np.median(sz, 0).round(2), 'L range', sz[:, 0].min().round(2), sz[:, 0].max().round(2))
print('points in box: median', int(np.median(npts)), 'share <5 pts', round(float(np.mean(np.array(npts) < 5)), 3))
