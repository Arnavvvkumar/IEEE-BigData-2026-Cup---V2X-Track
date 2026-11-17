"""Localisation FPs (best IoU 0.1-0.5 with a >=5-point pedestrian GT) among the top-3000 hedged-free PRIMARY pedestrian boxes:
which box component is limiting? Oracle per component (copy that component from the GT) -> share that reaches IoU >= 0.5.
Also signed errors (pred - GT): radial / tangential centre, bottom z, top z, height, length, width, |yaw| mod 90.
And the bottom z relative to the local ground (median z of ground points within 1.5 m), pred vs GT.
usage: ped_loc_diag.py <set dir> <norule pred name>"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
gt = {f: [np.array(g['box'], float) for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
pred = json.loads((d / f'pred_{name}.json').read_text())
P = {f: sorted([x for x in pred[f] if x['class'] == 'pedestrian'], key=lambda x: -x['score']) for f in ids}
allp = sorted(((x['score'], f, np.array(x['box'], float)) for f in ids for x in P[f]), key=lambda t: -t[0])[:3000]
clouds = {f: np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3).astype(float) for f in ids}


def ground_z(f, xy):
    q = clouds[f]
    q = q[np.abs(q).sum(1) > 0]
    gz = q[:, :2] @ calib.PLANE[:2] + calib.PLANE[2]
    m = (np.hypot(q[:, 0] - xy[0], q[:, 1] - xy[1]) < 1.5) & (np.abs(q[:, 2] - gz) < 0.15)
    return float(np.median(q[m, 2])) if m.sum() >= 5 else float(xy @ calib.PLANE[:2] + calib.PLANE[2])


rows, tps = [], []
for s, f, b in allp:
    if not gt[f]:
        continue
    ious = np.array([ev._iou_3d(b, g) for g in gt[f]])
    j = int(ious.argmax()); g = gt[f][j]; i0 = ious[j]
    if i0 >= 0.5:
        tps.append((f, b, g)); continue
    if i0 < 0.1:
        continue
    fix = {}
    for comp, sl in (('xy', slice(0, 2)), ('z', slice(2, 3)), ('h', slice(5, 6)), ('lw', slice(3, 5)), ('yaw', slice(6, 7)), ('z+h', None)):
        c = b.copy()
        if comp == 'z+h':
            c[2], c[5] = g[2], g[5]
        else:
            c[sl] = g[sl]
        fix[comp] = ev._iou_3d(c, g) >= 0.5
    u = g[:2] / max(np.hypot(*g[:2]), 1e-6); t = np.array([-u[1], u[0]])
    e = b[:2] - g[:2]
    rows.append(dict(i0=i0, fix=fix, er=e @ u, et=e @ t, bot=(b[2] - b[5] / 2) - (g[2] - g[5] / 2), top=(b[2] + b[5] / 2) - (g[2] + g[5] / 2),
                     h=b[5] - g[5], l=b[3] - g[3], w=b[4] - g[4], f=f, b=b, g=g))
print(f'{d.name}: top-3000 primaries | TP {len(tps)} | localisation FPs (IoU 0.1-0.5) {len(rows)}')
for comp in ('xy', 'z', 'h', 'z+h', 'lw', 'yaw'):
    print(f'  oracle {comp:4s}: {np.mean([r["fix"][comp] for r in rows]):.2f} of loc FPs reach IoU 0.5')
for k in ('er', 'et', 'bot', 'top', 'h', 'l', 'w'):
    v = np.array([r[k] for r in rows]); vt = None
    print(f'  {k:4s} loc-FP median {np.median(v):+.3f} mean {v.mean():+.3f} MAD {np.median(np.abs(v - np.median(v))):.3f}')
for k, fn in (('er', lambda b, g: (b[:2] - g[:2]) @ (g[:2] / np.hypot(*g[:2]))), ('bot', lambda b, g: (b[2] - b[5] / 2) - (g[2] - g[5] / 2)),
              ('top', lambda b, g: (b[2] + b[5] / 2) - (g[2] + g[5] / 2)), ('h', lambda b, g: b[5] - g[5]), ('l', lambda b, g: b[3] - g[3]), ('w', lambda b, g: b[4] - g[4])):
    v = np.array([fn(b, g) for _, b, g in tps])
    print(f'  {k:4s} TP     median {np.median(v):+.3f} mean {v.mean():+.3f}')
sub = rows[:400]
gb = np.array([(r['b'][2] - r['b'][5] / 2) - ground_z(r['f'], r['b'][:2]) for r in sub])
gg = np.array([(r['g'][2] - r['g'][5] / 2) - ground_z(r['f'], r['g'][:2]) for r in sub])
print(f'  bottom above local ground (first 400 loc FPs): pred median {np.median(gb):+.3f} MAD {np.median(np.abs(gb - np.median(gb))):.3f} | '
      f'GT median {np.median(gg):+.3f} MAD {np.median(np.abs(gg - np.median(gg))):.3f}')
print('  GT size median l w h', np.round(np.median([r['g'][3:6] for r in rows], 0), 3).tolist(), '| pred', np.round(np.median([r['b'][3:6] for r in rows], 0), 3).tolist())
