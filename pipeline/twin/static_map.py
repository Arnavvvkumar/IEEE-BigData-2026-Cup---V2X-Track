"""Site-twin step 1 (CPU, points only, no labels): how much of the real scene is static?
Levels each real scan with the median ground plane, voxelises at 0.25 m, and counts in how many scans each voxel is occupied.
Voxels occupied in >= FRAC of the scans form the static map. Reports static-point fraction per range band and saves the map.
Use of real data to disclose: unlabelled real clouds, used for site/sensor calibration only."""
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/erroranalysis'))
from groundplane import fit_ground, level_transform  # noqa: E402

KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
OUT = Path(__file__).parent
V, GZ = 0.25, -3.0
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
clouds = {f: np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
planes = np.array([fit_ground(clouds[f], np.random.default_rng(i)) for i, f in enumerate(ids)])
med = np.median(planes, 0)
R, t = level_transform(med, GZ)
lev = {f: (clouds[f] @ R.T + t) for f in ids}


def keys(p):
    return np.unique(np.floor(p / V).astype(np.int32), axis=0)


count = {}
for f in ids:
    for k in map(tuple, keys(lev[f])):
        count[k] = count.get(k, 0) + 1
n = len(ids)
print('frames', n, 'occupied voxels (any frame)', len(count))
for frac in (0.3, 0.5, 0.7):
    static = {k for k, c in count.items() if c >= frac * n}
    tot = st = 0
    bands = {'0-10': [0, 0], '10-20': [0, 0], '20-40': [0, 0], '40+': [0, 0]}
    for f in ids:
        p = lev[f]
        k = np.floor(p / V).astype(np.int32)
        hit = np.array([tuple(x) in static for x in k])
        r = np.hypot(p[:, 0], p[:, 1])
        tot += len(p)
        st += hit.sum()
        for name, lo, hi in (('0-10', 0, 10), ('10-20', 10, 20), ('20-40', 20, 40), ('40+', 40, 1e9)):
            m = (r >= lo) & (r < hi)
            bands[name][0] += int(m.sum())
            bands[name][1] += int(hit[m].sum())
    print(f'static if seen in >= {int(frac * 100)}% of scans: {len(static)} voxels, {100 * st / tot:.1f}% of all points static;',
          {k: round(100 * v[1] / max(v[0], 1), 1) for k, v in bands.items()}, flush=True)
    if frac == 0.5:
        np.save(OUT / 'static_voxels_v025_g-3.0.npy', np.array(sorted(static), dtype=np.int32))
np.save(OUT / 'median_plane.npy', med)
print('median plane', med.round(4).tolist())
