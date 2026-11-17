"""Check that final/fuse_test.py's fusion reproduces canon/fuse.py's adopted dev output (fuse_cw_b_ep10, mean r0.6) exactly."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fuse_test as FT  # noqa: E402

F = FT.W / 'experiments/claude_carla_overnight/canon/fusion'
spec = [('vnx010_all_s1234_ep10', 'pedestrian,vehicle'), ('canon_all_s1234_ep12', 'vehicle,truck'), ('canon_all_rs2_ep12', 'vehicle,truck'),
        ('canon_relabel_s1234_ep12', 'vehicle,truck')]
passes = []
for name, keep in spec:
    data = json.loads((F / f'{name}_allpred.json').read_text())
    kc = set(keep.split(','))
    passes += [{f: [d for d in p[f] if d['class'] in kc] for f in p} for p in data.values()]
ref = json.loads((F / 'fuse_cw_b_ep10_predictions.json').read_text())
print('best mode in reference run:', json.loads((F / 'fuse_cw_b_ep10.json').read_text())['best_by_selection'])
mine = FT.fuse(passes, list(ref))
worst = 0.0
for f in ref:
    assert len(ref[f]) == len(mine[f]), f
    for a, b in zip(ref[f], mine[f]):
        assert a['class'] == b['class']
        worst = max(worst, float(np.abs(np.array(a['box']) - np.array(b['box'])).max()), abs(a['score'] - b['score']))
print('frames', len(ref), 'max abs difference', worst)
