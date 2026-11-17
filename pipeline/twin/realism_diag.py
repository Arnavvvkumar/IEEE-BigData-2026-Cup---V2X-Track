"""Realism diagnosis (dev frames only; the released test scans are only counted, never used for scoring/selection):
 1) are the submitted synthetic scans the ones from twin/final_synthetic_v1?
 2) share of points at the origin (no-return zeros) inside the scoring box: submitted synthetic vs dev frames vs (count only) test scans
 3) local kit realism (lean, same maths as the scorer) of the submitted set vs the dev SELECTION and CONFIRMATION halves,
    and the same with the zero count per scan scaled by k (zeros removed or added), to see if zero share drives EMD.
usage: realism_diag.py"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import range_bg as T  # noqa: E402

W = T.W
SUB = W / 'experiments/claude_carla_overnight/final/candidate_v3/synthetic'
FS1 = HERE / 'final_synthetic_v1/synthetic'
md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
same = all(md5(SUB / f.name) == md5(f) for f in sorted(FS1.glob('*.bin')))
print('submitted synthetic == final_synthetic_v1:', same, flush=True)
syn = [np.fromfile(f, dtype='<f4').reshape(-1, 3).astype(np.float64) for f in sorted(SUB.glob('*.bin'))]
B = (-40, -40, -8, 40, 40, 2)
inbox = lambda p: p[(p[:, 0] >= B[0]) & (p[:, 0] <= B[3]) & (p[:, 1] >= B[1]) & (p[:, 1] <= B[4]) & (p[:, 2] >= B[2]) & (p[:, 2] <= B[5])]


def zshare(clouds):
    z = n = 0
    for p in clouds:
        q = inbox(p)
        z += int((np.abs(q).sum(1) == 0).sum()); n += len(q)
    return z / n, n / len(clouds)


SEL = [T.clouds[f] for f in T.split['selection_ids']]
CONF = [T.clouds[f] for f in T.split['confirmation_ids']]
test = [np.fromfile(f, dtype='<f4').reshape(-1, 3) for f in sorted((W / 'starting_kit/track_b/detection_test_frames').glob('*.bin'))]
for name, c in (('submitted synthetic', syn), ('dev selection half', SEL), ('dev confirmation half', CONF), ('released test scans (count only)', test)):
    s, n = zshare(c)
    print(f'{name:34s} in-box points/scan {n:8.0f}  zero share {100 * s:5.1f}%', flush=True)


def scale_zeros(clouds, k, rng):
    out = []
    for p in clouds:
        z = np.abs(p).sum(1) == 0
        nz = int(round(z.sum() * k))
        out.append(np.vstack([p[~z], np.zeros((nz, 3))]))
    return out


t = time.time()
base_sel = T.lean(syn, SEL)
print(f'lean() takes {time.time() - t:.0f} s', flush=True)
print('submitted vs SEL :', base_sel, flush=True)
print('submitted vs CONF:', T.lean(syn, CONF), flush=True)
rng = np.random.default_rng(0)
for k in (0.8, 0.9, 0.95, 1.05, 1.15):
    s = scale_zeros(syn, k, rng)
    r1, r2 = T.lean(s, SEL), T.lean(s, CONF)
    print(f'zeros x{k:4.2f}: SEL realism {r1["realism"]} EMD_n {r1.get("EMD_n")} | CONF realism {r2["realism"]} EMD_n {r2.get("EMD_n")}', flush=True)
print('REALISM_DIAG_DONE')
