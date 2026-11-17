"""Real scans encode no-return beams as points at exactly (0,0,0) (~15.6k of 65,536 per scan), and the kit range mask keeps them.
Test what emitting no-return beams the same way does to the realism score of (a) the range-image twin, (b) the CARLA v3 scans.
Cross-fitted for the twin (build on one half, score on the other)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import range_bg as T  # noqa: E402

W = T.W
res = {}
kit = sorted((W / 'starting_kit/track_b/detection_test_frames').glob('*.bin'))
z = [int((np.linalg.norm(np.fromfile(f, dtype='<f4').reshape(-1, 3), axis=1) == 0).sum()) for f in kit]
n = [len(np.fromfile(f, dtype='<f4').reshape(-1, 3)) for f in kit]
print('released test scans:', len(kit), 'points/scan median', int(np.median(n)), 'exact-zero median', int(np.median(z)), flush=True)
res['released_test_zero_median'] = int(np.median(z))
sel, conf = T.split['selection_ids'], T.split['confirmation_ids']
nz = {f: int((np.linalg.norm(T.clouds[f], axis=1) == 0).sum()) for f in T.ids}


def with_zeros(scan, k):
    return np.concatenate([scan, np.zeros((k, 3), np.float32)])


for name, a, b in (('twin_zeros_build_sel_score_conf', sel, conf), ('twin_zeros_build_conf_score_sel', conf, sel)):
    model = T.build(a)
    rng = np.random.default_rng(7)
    zs = np.array([nz[f] for f in a])
    scans = [with_zeros(T.sample(model, rng), int(rng.choice(zs))) for _ in range(50)]
    res[name] = {**T.lean(scans, [T.clouds[f] for f in b]), 'points_per_scan': int(np.median([len(s) for s in scans]))}
    print(name, res[name], flush=True)
sv3 = json.loads((W / 'experiments/claude_carla_overnight/realism/realism_select_v3.json').read_text())['v3_greedy']['frames']
D = Path.home() / 'projects/bdc2026-track-b/data/claude_carla_v3/tilted'
v3 = [np.load(D / f'{f}.npy').astype(np.float32) for f in sv3]
rng = np.random.default_rng(7)
zs = np.array(list(nz.values()))
res['v3_greedy_plain'] = T.lean(v3, [T.clouds[f] for f in T.ids])
print('v3_greedy_plain', res['v3_greedy_plain'], flush=True)
res['v3_greedy_zeros'] = T.lean([with_zeros(s, int(rng.choice(zs))) for s in v3], [T.clouds[f] for f in T.ids])
print('v3_greedy_zeros', res['v3_greedy_zeros'], 'v3 points/scan', int(np.median([len(s) for s in v3])), flush=True)
(HERE / 'zeros_test_v1.json').write_text(json.dumps(res, indent=1))
