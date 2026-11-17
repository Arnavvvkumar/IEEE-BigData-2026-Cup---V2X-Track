"""Apply the synthetic-trained refiner to the pedestrian boxes of a finished prediction file on a validation set and score
pedestrian AP (all / half A / half B) for several ways of using its output. Labels are only used for scoring.
usage: eval_real.py <set dir> <pred name> <refiner .pt>"""
import importlib.util
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import fuse_test as FT  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
halves = [ids, ids[0::2], ids[1::2]]
gt = {f: [{'class': 'pedestrian', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
pred = {f: [x for x in v if x['class'] == 'pedestrian'] for f, v in json.loads((d / f'pred_{name}.json').read_text()).items()}


def refine_all():
    net = C.build_net()
    net.load_state_dict(torch.load(sys.argv[3], map_location='cpu'))
    net.eval()
    clouds = FT.load(d / 'scans')
    rng = np.random.default_rng(0)
    out = {}
    for f in ids:
        p = np.asarray(clouds[f])[:, :3].astype(np.float64)
        p = p[np.abs(p).sum(1) > 0] @ C.R.T + C.T
        X, A, U, I = [], [], [], []
        for i, x in enumerate(pred[f]):
            b = np.array(x['box'], float)
            c = C.R @ b[:3] + C.T
            near = p[(np.abs(p[:, 0] - c[0]) < 1.2) & (np.abs(p[:, 1] - c[1]) < 1.2)]
            r = C.crop(near, np.r_[c, b[3:]], rng)
            if r is not None and r[3] >= 5:
                X.append(r[0]); A.append(r[1]); U.append(r[2]); I.append(i)
        delta = {}
        if X:
            with torch.no_grad():
                o = net(torch.from_numpy(np.array(X)), torch.from_numpy(np.array(A))).numpy()
            for i, u, v in zip(I, U, o):
                delta[i] = (np.array([v[0] * u[0] - v[1] * u[1], v[0] * u[1] + v[1] * u[0], v[2]]), float(v[3]), u)
        out[f] = delta
    return out


def variant(job):
    label, wxy, wz, wh = job[:4]
    bias = job[4] if len(job) > 4 else 0.0   # extra shift along the sensor ray (m, + = away from the sensor)
    new = {}
    for f in ids:
        o = []
        for i, x in enumerate(pred[f]):
            b = list(x['box'])
            if i in DELTA[f]:
                dc, dh, u = DELTA[f][i]
                c = C.R @ np.array(b[:3]) + C.T + np.array([wxy * dc[0] + bias * u[0], wxy * dc[1] + bias * u[1], wz * dc[2]])
                b[:3] = (C.R.T @ (c - C.T)).tolist()
                b[5] = b[5] + wh * dh
            o.append({**x, 'box': b})
        new[f] = o
    res = []
    for h in halves:
        m = ev.evaluate_3d_detection({f: new[f] for f in h}, {f: gt[f] for f in h}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        res.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 1))
    return f'{label:34s} {res}'


if __name__ == '__main__':
    DELTA = refine_all()
    n = sum(len(v) for v in DELTA.values())
    sh = np.array([np.r_[v[0], v[1]] for dl in DELTA.values() for v in dl.values()])
    print(f'refined {n} of {sum(len(v) for v in pred.values())} boxes | mean |shift| x y z h: {np.abs(sh).mean(0).round(3).tolist()}', flush=True)
    jobs = [('unchanged', 0, 0, 0), ('centre xy, full', 1, 0, 0), ('centre xy, half', 0.5, 0, 0), ('centre xy + z, full', 1, 1, 0),
            ('centre xy + z + height, full', 1, 1, 1), ('centre xy half + z half', 0.5, 0.5, 0), ('z + height only', 0, 1, 1)]
    if len(sys.argv) > 4 and sys.argv[4] == 'bias':   # does the refiner only disagree with the real labels by a constant ray offset?
        sh3 = np.array([v[0][:2] @ v[2] for dl in DELTA.values() for v in dl.values()])
        print(f'mean refiner shift along the sensor ray: {sh3.mean():+.3f} m (+ = away from sensor)', flush=True)
        jobs = [('unchanged', 0, 0, 0)] + [(f'centre xy full, ray bias {b:+.2f}', 1, 0, 0, b) for b in (-0.12, -0.08, -0.04, 0.04, 0.08, 0.12)] \
            + [(f'centre xy half, ray bias {b:+.2f}', 0.5, 0, 0, b) for b in (-0.06, -0.03, 0.03, 0.06)]
    with get_context('fork').Pool(6) as pool:
        for line in pool.imap(variant, jobs):
            print(line, flush=True)
    print('REFINER_EVAL_DONE')
