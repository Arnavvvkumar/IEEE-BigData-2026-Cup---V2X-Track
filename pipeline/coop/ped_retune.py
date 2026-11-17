"""Re-tune pedestrian hedging on the CURRENT ensemble (PED_HEDGE was set on a v3-era ensemble at AP 38.5; now ~45.4).
Pre-registered grid. Pedestrian AP only (hedging touches nothing else). Halves reported.
usage: ped_retune.py <set dir> <norule pred name> [workers]"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
gt = {f: [{'class': 'pedestrian', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
pred = {f: [x for x in v if x['class'] == 'pedestrian'] for f, v in json.loads((d / f'pred_{name}.json').read_text()).items()}
H0 = calib.PED_HEDGE


def radial(p, shifts, k, smin=0.05):
    """extra variants moved along the sensor ray by each shift (m) at k x score"""
    out = {}
    for f, dets in p.items():
        new = list(dets)
        for x in dets:
            if x['score'] >= smin:
                b = x['box']; r = max(np.hypot(b[0], b[1]), 1e-6)
                for s in shifts:
                    new.append({**x, 'box': [b[0] + s * b[0] / r, b[1] + s * b[1] / r, *b[2:]], 'score': x['score'] * k})
        out[f] = new
    return out


def scaled(h, m):
    return [(a, b, c, e, min(1.0, k * m)) for a, b, c, e, k in h]


def ap(p):
    out = []
    for h in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: gt[f] for f in h}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 2))
    return out


DIAG2 = [(1, 1, a, b, 0.1) for a in (0.2, -0.2) for b in (0.2, -0.2)]
MID = [(0.9, 0.9, 0, 0, 0.3), (1.1, 1.1, 0, 0, 0.3)]
JOBS = [
    ('current', dict(h=H0)),
    ('factors x0.5', dict(h=scaled(H0, 0.5))),
    ('factors x1.5', dict(h=scaled(H0, 1.5))),
    ('factors x2', dict(h=scaled(H0, 2.0))),
    ('smin 0.02', dict(h=H0, smin=0.02)),
    ('smin 0.1', dict(h=H0, smin=0.1)),
    ('+ mid sizes .9/1.1 f.3', dict(h=H0 + MID)),
    ('+ diag shifts 0.2', dict(h=H0 + DIAG2)),
    ('+ radial +-0.15 f.1', dict(h=H0, rad=((0.15, -0.15), 0.1))),
    ('+ radial +-0.15 f.2', dict(h=H0, rad=((0.15, -0.15), 0.2))),
    ('+ radial +-0.1,0.25 f.1', dict(h=H0, rad=((0.1, -0.1, 0.25, -0.25), 0.1))),
    ('+ radial away 0.15 f.2', dict(h=H0, rad=((0.15,), 0.2))),
    ('+ radial toward 0.15 f.2', dict(h=H0, rad=((-0.15,), 0.2))),
]


def run(job):
    label, kw = job
    p = pred
    if 'rad' in kw:
        p = radial(p, *kw['rad'])
        # radial variants are themselves not re-hedged: hedge only the originals, then add radial ones
        base = calib.hedge_pedestrians(pred, kw['h'], kw.get('smin', 0.05))
        extra = {f: p[f][len(pred[f]):] for f in ids}
        p = {f: base[f] + extra[f] for f in ids}
    else:
        p = calib.hedge_pedestrians(p, kw['h'], kw.get('smin', 0.05))
    return f'{d.name} {label:26s} ped AP {ap(p)} | boxes/frame {sum(len(v) for v in p.values()) / len(ids):.0f}'


if __name__ == '__main__':
    from multiprocessing import get_context
    with get_context('fork').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        for line in pool.imap(run, JOBS):
            print(line, flush=True)
    print('PED_RETUNE_DONE')
