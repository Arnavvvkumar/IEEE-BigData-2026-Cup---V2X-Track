"""Vehicle box hedging (never tried before; pedestrian hedging gave +4 AP). Emit low-score size/position/yaw variants of each
vehicle box, primary boxes unchanged. Four pre-registered variant sets, evaluated on dev150 and dev350 with halves.
usage: veh_hedge_exp.py <set dir> <norule pred name>"""
import importlib.util, json, sys
import numpy as np
from pathlib import Path

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
gt = {f: [{'class': 'vehicle', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'vehicle'] for f in ids}
pred = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))

# (length scale, width scale, shift along heading m, shift across m, yaw delta rad, score factor)
SETS = {
    'none': [],
    'size4': [(0.9, 0.9, 0, 0, 0, 0.1), (1.1, 1.1, 0, 0, 0, 0.1), (0.85, 1.0, 0, 0, 0, 0.05), (1.15, 1.0, 0, 0, 0, 0.05)],
    'shift4': [(1, 1, 0.4, 0, 0, 0.1), (1, 1, -0.4, 0, 0, 0.1), (1, 1, 0, 0.25, 0, 0.05), (1, 1, 0, -0.25, 0, 0.05)],
    'yaw2': [(1, 1, 0, 0, 0.12, 0.1), (1, 1, 0, 0, -0.12, 0.1)],
    'mix8': [(0.9, 0.9, 0, 0, 0, 0.1), (1.1, 1.1, 0, 0, 0, 0.1), (1, 1, 0.4, 0, 0, 0.1), (1, 1, -0.4, 0, 0, 0.1),
             (1, 1, 0, 0.25, 0, 0.05), (1, 1, 0, -0.25, 0, 0.05), (1, 1, 0, 0, 0.12, 0.05), (1, 1, 0, 0, -0.12, 0.05)],
}

def hedge(p, variants, smin=0.05):
    out = {}
    for f, dets in p.items():
        new = list(dets)
        for x in dets:
            if x['class'] == 'vehicle' and x['score'] >= smin:
                b = x['box']; c, s = np.cos(b[6]), np.sin(b[6])
                for sl, sw, da, dc, dyaw, k in variants:
                    new.append({**x, 'box': [b[0] + da * c - dc * s, b[1] + da * s + dc * c, b[2], b[3] * sl, b[4] * sw, b[5], b[6] + dyaw],
                                'score': x['score'] * k})
        out[f] = new
    return out

for label, v in SETS.items():
    p = hedge(pred, v)
    q = {f: [x for x in p[f] if x['class'] == 'vehicle'] for f in ids}
    out = []
    for h in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: q[f] for f in h}, {f: gt[f] for f in h}, ['vehicle'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['vehicle']['AP_0.5_R40'], 2))
    print(f'{d.name} {label:8s} vehicle AP {out} boxes/frame {sum(len(v) for v in q.values()) / len(ids):.0f}', flush=True)
