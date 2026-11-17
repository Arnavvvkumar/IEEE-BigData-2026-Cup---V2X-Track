"""What does the 4th (intensity) channel of the synthetic training frames contain? Samples 40 frames per training set."""
from pathlib import Path

import numpy as np

D = Path.home() / 'projects/bdc2026-track-b/data'
for s in ('claude_canon_all_v1', 'claude_twin_v1', 'claude_twin_st_v1', 'claude_twin_pv_v1', 'claude_twin_ll_v1'):
    fs = sorted((D / s / 'points').glob('*.npy'))[::max(1, len(list((D / s / 'points').glob('*.npy'))) // 40)][:40]
    a = np.concatenate([np.load(f)[:, 3] for f in fs])
    u = np.unique(np.round(a, 4))
    print(f'{s:22s} intensity: min {a.min():.3f} max {a.max():.3f} mean {a.mean():.3f} distinct values {len(u)} | share == 0: {100 * (a == 0).mean():.1f}%')
