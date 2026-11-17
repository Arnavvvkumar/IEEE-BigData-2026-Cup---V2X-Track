"""Check a converted v3 dataset: ground plane of tilted and level clouds, elevation span, box hits, class counts."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'erroranalysis'))
from groundplane import fit_ground  # noqa: E402

d = Path(sys.argv[1])
recs = json.loads((d / 'frames.json').read_text())
step = max(1, len(recs) // 12)
for r in recs[::step]:
    t = np.load(d / 'tilted' / f"{r['frame']}.npy")
    p = np.load(d / 'points' / f"{r['frame']}.npy")
    pt, pl = fit_ground(t, np.random.default_rng(0)), fit_ground(p[:, :3], np.random.default_rng(0))
    el = np.degrees(np.arcsin(t[:, 2] / np.maximum(np.linalg.norm(t, axis=1), 1e-6)))
    hits = [o['hits'] for o in r['object_details'] if o['annotated']]
    peds = [o['hits'] for o in r['object_details'] if o['annotated'] and o['class'] == 'pedestrian']
    print(r['frame'], r['map'].split('/')[-1], 'tilted plane', pt.round(3).tolist(), 'level plane', pl.round(3).tolist(), 'pts', len(p),
          'share<10m', round(float((np.hypot(t[:, 0], t[:, 1]) < 10).mean()), 2), 'el', np.percentile(el, [0.5, 99.5]).round(1).tolist(),
          r['class_counts'], 'ped hits median', int(np.median(peds)) if peds else None)
print('real plane target: [-0.006, -0.113, -2.87]; real share<10m 0.53; real pts ~50k; real ped points median 18')
