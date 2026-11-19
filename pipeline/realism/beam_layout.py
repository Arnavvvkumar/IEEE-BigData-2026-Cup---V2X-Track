"""Sensor calibration from the public real scans (points only): elevation-angle beam layout, azimuth resolution, range stats.
Used to set the CARLA sensor profile (channels, upper/lower FOV, points per second)."""
import os
import json
from pathlib import Path

import numpy as np

KIT = Path(os.environ['BDC_ROOT'] + '/experiments/real_reference_audit/real_dev_cw1')
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
out = {}
els, ns, rmax = [], [], []
for f in ids[:50]:
    p = np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3)
    r = np.linalg.norm(p, axis=1)
    p, r = p[r > 0.5], r[r > 0.5]
    els.append(np.degrees(np.arcsin(p[:, 2] / r)))
    ns.append(len(p)); rmax.append(float(np.percentile(r, 99.9)))
el = np.concatenate(els)
out['points_per_scan_all_ranges_median'] = int(np.median(ns))
out['range_p99.9_median'] = round(float(np.median(rmax)), 1)
out['elevation_deg_percentiles_0.1_1_5_25_50_75_95_99_99.9'] = np.percentile(el, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]).round(2).tolist()
h, e = np.histogram(el, bins=np.arange(-60, 60.05, 0.1))
# beam peaks: local maxima above 20% of the median non-zero bin
thr = 0.5 * np.median(h[h > 0])
peaks = [round(float((e[i] + e[i + 1]) / 2), 2) for i in range(1, len(h) - 1) if h[i] > thr and h[i] >= h[i - 1] and h[i] > h[i + 1]]
out['n_beam_peaks_0.1deg_bins'] = len(peaks)
out['beam_peaks_deg'] = peaks
if len(peaks) > 2:
    d = np.diff(peaks)
    out['beam_spacing_deg_median_min_max'] = [round(float(np.median(d)), 2), round(float(d.min()), 2), round(float(d.max()), 2)]
# azimuth resolution on the densest beam of the first frame
p = np.fromfile(KIT / 'detection_test_frames' / f'{ids[0]}.bin', dtype='<f4').reshape(-1, 3)
r = np.linalg.norm(p, axis=1)
elv = np.degrees(np.arcsin(p[:, 2] / np.maximum(r, 1e-6)))
best = None
for pk in peaks:
    m = np.abs(elv - pk) < 0.08
    if best is None or m.sum() > best[1]:
        best = (pk, int(m.sum()), m)
az = np.sort(np.degrees(np.arctan2(p[best[2], 1], p[best[2], 0])))
da = np.diff(az)
out['densest_beam'] = {'elevation': best[0], 'points': best[1], 'azimuth_step_deg_median': round(float(np.median(da[da > 1e-3])), 4)}
out['coarse_elevation_hist_5deg'] = {f'{int(a)}..{int(a + 5)}': int(c) for a, c in zip(np.arange(-60, 60, 5), np.histogram(el, np.arange(-60, 65, 5))[0]) if c}
Path(__file__).with_name('beam_layout_real.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
