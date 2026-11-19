"""How do the 50 test scans (50,000 points) differ from the validation scans (65,536 points)?
Compares non-zero point share by azimuth sector, elevation band and range band, and where zeros sit in file order.
usage: density_shape.py <dev set dir>"""
import os
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import fuse_test as FT  # noqa: E402

T = list(FT.load(W / 'starting_kit/track_b/detection_test_frames').values())
Dv = list(FT.load(Path(sys.argv[1]) / 'scans').values())[:50]


def hist(clouds, fn, bins):
    h = np.zeros(len(bins) - 1)
    for p in clouds:
        p = np.asarray(p)[:, :3]
        p = p[np.abs(p).sum(1) > 0]
        h += np.histogram(fn(p), bins)[0]
    return h / len(clouds)


az = lambda p: np.degrees(np.arctan2(p[:, 1], p[:, 0]))
el = lambda p: np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))
rg = lambda p: np.linalg.norm(p, axis=1)
for label, fn, bins in [('azimuth deg', az, np.arange(-180, 181, 30)), ('elevation deg', el, np.array([-90, -40, -30, -20, -15, -10, -5, 0, 5, 20, 90])),
                        ('range m', rg, np.array([0, 5, 10, 15, 20, 30, 40, 60, 100, 300]))]:
    t, d = hist(T, fn, bins), hist(Dv, fn, bins)
    print(f'--- non-zero points per scan by {label}: test / dev = ratio')
    for i in range(len(bins) - 1):
        print(f'   {bins[i]:6.0f}..{bins[i + 1]:6.0f}: {t[i]:7.0f} / {d[i]:7.0f} = {t[i] / max(d[i], 1):.2f}')
for label, cl in (('test', T), ('dev', Dv)):
    p = np.asarray(cl[0])[:, :3]
    z = np.abs(p).sum(1) == 0
    q = len(p) // 8
    print(f'{label}: first scan {len(p)} points; zero share by eighth of file order: ' + ' '.join(f'{z[i * q:(i + 1) * q].mean():.2f}' for i in range(8)))
    e = el(p[~z])
    print(f'   distinct elevation rows (0.1 deg bins with >50 pts): {(np.histogram(e, np.arange(-90, 30, 0.1))[0] > 50).sum()}')
    # is the point order structured? correlation of consecutive azimuths
    a = az(p[~z][:20000])
    print(f'   median |azimuth step| between consecutive points: {np.median(np.abs(np.diff(a))):.3f} deg')
