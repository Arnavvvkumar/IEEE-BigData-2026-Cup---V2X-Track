"""final_synthetic_v4: the submitted 50 synthetic scans (final_synthetic_v1) with (1) non-zero points thinned by RANGE so their range histogram
matches the real development frames (keep probability ~ real share / synthetic share per range band, fitted on ALL unlabelled dev frames) and
(2) the no-return zero count set so zeros are 30.8% of the points INSIDE the scoring box (the real dev share). Non-zero points are only removed,
never moved or created. Dev frames only; the released test scans are never read. Cross-fitted estimate: realism +0.005 to +0.0065 (5 seeds).
usage: make_synthetic_v4.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import range_bg as T  # noqa: E402

RB = np.array([0, 5, 10, 15, 20, 25, 30, 40, 57, 1e9])
BOX = (-40, -40, -8, 40, 40, 2)
SHARE = 0.308
syn_files = sorted((HERE / 'final_synthetic_v1/synthetic').glob('*.bin'))
syn = [np.fromfile(f, dtype='<f4').reshape(-1, 3).astype(np.float64) for f in syn_files]
dev_all = [T.clouds[f] for f in T.split['selection_ids'] + T.split['confirmation_ids']]
SEL = [T.clouds[f] for f in T.split['selection_ids']]
CONF = [T.clouds[f] for f in T.split['confirmation_ids']]
nz = lambda cl: (lambda p: p[np.abs(p).sum(1) > 0])(np.concatenate([np.asarray(c, np.float64) for c in cl]))
band = lambda p: np.clip(np.digitize(np.hypot(p[:, 0], p[:, 1]), RB) - 1, 0, len(RB) - 2)
inbox = lambda q: (q[:, 0] >= BOX[0]) & (q[:, 0] <= BOX[3]) & (q[:, 1] >= BOX[1]) & (q[:, 1] <= BOX[4]) & (q[:, 2] >= BOX[2]) & (q[:, 2] <= BOX[5])

r, s = nz(dev_all), nz(syn)
n = len(RB) - 1
hr, hs, cnt = np.bincount(band(r), minlength=n) / len(r), np.bincount(band(s), minlength=n) / len(s), np.bincount(band(s), minlength=n)
ratio = np.where(cnt >= 2000, hr / np.maximum(hs, 1e-12), np.nan)
keep = np.where(np.isnan(ratio), 1.0, np.clip(ratio / np.nanmax(ratio), 0.3, 1.0))
print('range bands (m)      :', RB[:-1].tolist())
print('real share           :', np.round(hr, 4).tolist())
print('synthetic share      :', np.round(hs, 4).tolist())
print('keep probability     :', np.round(keep, 3).tolist(), flush=True)

rng = np.random.default_rng(20261113)
out = HERE / 'final_synthetic_v4/synthetic'
out.mkdir(parents=True, exist_ok=True)
scans = []
for f, p in zip(syn_files, syn):
    q = p[np.abs(p).sum(1) > 0]
    q = q[rng.random(len(q)) < keep[band(q)]]
    z = np.zeros((int(round(SHARE / (1 - SHARE) * int(inbox(q).sum()))), 3))
    a = np.vstack([q, z]).astype('<f4')
    a.tofile(out / f.name)
    scans.append(a.astype(np.float64))
back = [np.fromfile(f, dtype='<f4').reshape(-1, 3) for f in sorted(out.glob('*.bin'))]
assert len(back) == 50 and max(f.stat().st_size for f in out.glob('*.bin')) <= 1_500_000
zs = [(np.abs(b).sum(1) == 0).sum() / int(inbox(b.astype(np.float64)).sum()) for b in back]
print('wrote', len(back), 'scans | max bytes', max(f.stat().st_size for f in out.glob('*.bin')), '| median points', int(np.median([len(b) for b in back])),
      '| zero share inside box', round(float(np.mean(zs)), 3), flush=True)
res = {}
for h, ref in (('SEL', SEL), ('CONF', CONF)):
    r5 = [T.lean(scans, ref, seed=k) for k in (1234, 1, 2, 3, 4)]
    res[h] = {'seed1234': r5[0], 'mean5': round(float(np.mean([x['realism'] for x in r5])), 4), 'sd5': round(float(np.std([x['realism'] for x in r5])), 4)}
    print(h, '| seed 1234:', r5[0], '| 5-seed mean', res[h]['mean5'], 'sd', res[h]['sd5'], '(ratios fitted on all dev frames, so slightly optimistic; cross-fitted estimate is in realism_exp6b.log)', flush=True)
(out.parent / 'selection_log.json').write_text(json.dumps({'share': SHARE, 'range_bins': RB[:-1].tolist(), 'keep': keep.tolist(), 'result': res}, indent=1))
