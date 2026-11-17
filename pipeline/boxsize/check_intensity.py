"""Every training/val cloud of the relabelled mix has an all-zero intensity column (so a zero feature matches real eval padding)."""
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
bad, n = [], 0
for split in ('train', 'val'):
    for f in (d / 'ImageSets' / f'{split}.txt').read_text().split():
        p = np.load(d / 'points' / f'{f}.npy', mmap_mode='r')
        n += 1
        if p.shape[1] != 4 or np.any(p[:, 3] != 0):
            bad.append(f)
print({'frames': n, 'nonzero_intensity_frames': len(bad), 'examples': bad[:5]})
