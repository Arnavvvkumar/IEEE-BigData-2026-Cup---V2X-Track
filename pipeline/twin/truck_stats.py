"""Truck box sizes: real validation labels by type (box statistics only) vs the synthetic training sets.
usage: truck_stats.py"""
import json
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight')
D = Path.home() / 'projects/bdc2026-track-b/data'
q = lambda a: ' '.join(f'{x:5.2f}' for x in np.percentile(a, [10, 50, 90]))
G = json.loads((W / 'coop/dev350/gt_coop.json').read_text())
for t in ('Bus', 'Truck', 'Van', 'TrashCan'):
    b = np.array([g['box'] for v in G.values() for g in v if g['class'] == 'truck' and g.get('type') == t and g['points'] >= 5])
    if len(b):
        print(f'real {t:9s} n {len(b):4d} | L p10/50/90 {q(b[:, 3])} | W {q(b[:, 4])} | H {q(b[:, 5])} | range {q(np.hypot(b[:, 0], b[:, 1]))} | points {q(np.array([g["points"] for v in G.values() for g in v if g["class"] == "truck" and g.get("type") == t and g["points"] >= 5]))}')
for s in ('claude_canon_all_v1', 'claude_twin_v1', 'claude_twin_st_v1', 'claude_twin_pv_v1'):
    rows, frames, nf = [], 0, 0
    files = sorted((D / s / 'labels').glob('*.txt'))
    for f in files[::max(1, len(files) // 1500)]:
        nf += 1
        k = 0
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) >= 8 and p[7] == 'truck':
                rows.append([float(x) for x in p[:7]])
                k += 1
        frames += k > 0
    b = np.array(rows)
    print(f'synth {s:22s} trucks/frame {len(b) / nf:4.2f} frames with a truck {100 * frames / nf:3.0f}% | L {q(b[:, 3])} | W {q(b[:, 4])} | H {q(b[:, 5])} | range {q(np.hypot(b[:, 0], b[:, 1]))} | share L>=7.5 & H<=2.9: {100 * ((b[:, 3] >= 7.5) & (b[:, 5] <= 2.9)).mean():.0f}%')
