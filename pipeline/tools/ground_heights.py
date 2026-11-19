"""Ground height in the sensor frame: Generator V2 metadata + cloud percentile, real reference clouds, CARLA dry run."""
import os
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
recs = json.loads((Path.home() / 'projects/bdc2026-track-b/data/claude_v2_scale5500_fix1/frames.json').read_text())
h = np.array([r['sensor']['height_m'] for r in recs if r['split'] == 'train'])
print('GenV2 sensor height_m p5/50/95', np.percentile(h, [5, 50, 95]).round(2), 'min/max', h.min().round(2), h.max().round(2))


def ground(p):
    r = np.hypot(p[:, 0], p[:, 1])
    m = (r > 4) & (r < 20)
    return float(np.percentile(p[m, 2], 10)) if m.sum() > 100 else None


pcd = Path.home() / 'projects/bdc2026-track-b/data/codex_openpcdet_custom_v1/points'
files = sorted(pcd.glob('*.npy'))[::100]
g = [ground(np.load(f)) for f in files]
print('GenV2 ground z p10(4-20m) median', np.median([x for x in g if x is not None]).round(2), 'p5/p95', np.percentile([x for x in g if x is not None], [5, 95]).round(2))
ref = W / 'experiments/real_reference_audit/train_reference50_verified'
gr = []
for b in sorted(ref.rglob('*.bin')):
    p = np.fromfile(b, dtype='<f4').reshape(-1, 4)
    p = p[np.isfinite(p[:, :3]).all(1)]
    gr.append(ground(p))
print('Real ground z p10(4-20m) median', np.median(gr).round(2), 'p5/p95', np.percentile(gr, [5, 95]).round(2))
c = W / 'experiments/claude_carla_overnight/dryrun_converted/points'
gc = [ground(np.load(f)) for f in sorted(c.glob('*.npy'))[::20]]
print('CARLA ground z p10(4-20m) median', np.median(gc).round(2), 'p5/p95', np.percentile(gc, [5, 95]).round(2))
