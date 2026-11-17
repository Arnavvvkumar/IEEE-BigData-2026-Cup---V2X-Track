"""Format + sanity check of test predictions against the kit's frame list, and comparison of detection statistics with the
dev-frame output of the same ensemble (no labels involved). usage: check_candidate.py <predictions.json>"""
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
pred = json.loads(Path(sys.argv[1]).read_text())
ids = json.loads((W / 'starting_kit/track_b/detection_test_frame_ids.json').read_text())
assert set(pred) == set(ids) and len(ids) == 50, (len(pred), len(ids))
C = ['vehicle', 'pedestrian', 'truck']
for f, dets in pred.items():
    for d in dets:
        assert set(d) == {'box', 'score', 'class'} and d['class'] in C and len(d['box']) == 7
        assert np.isfinite(d['box']).all() and 0 <= d['score'] <= 1 and min(d['box'][3:6]) > 0
dev = json.loads((W / 'experiments/claude_carla_overnight/twin/static_filter_cw_d_predictions.json').read_text())


def stats(p):
    row = {}
    for c in C:
        s = np.array([d['score'] for v in p.values() for d in v if d['class'] == c])
        row[c] = {'per_frame': round(len(s) / len(p), 1), 'score>=0.3': round(float((s >= 0.3).sum()) / len(p), 2),
                  'score>=0.5': round(float((s >= 0.5).sum()) / len(p), 2)}
    return row


print('format OK, frames', len(pred), 'bytes', Path(sys.argv[1]).stat().st_size)
print('test', json.dumps(stats(pred)))
print('dev ', json.dumps(stats(dev)))
