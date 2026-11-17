"""Pedestrian hedge with HEIGHT variants (current PED_HEDGE only scales length/width). Pre-registered sets, dev150/dev350 with halves.
usage: ped_hhedge_exp.py <set dir> <norule pred name>"""
import importlib.util, json, sys
from pathlib import Path

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
gt = {f: [{'class': 'pedestrian', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
pred = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))
base = calib.hedge_pedestrians(pred)

def hh(p, variants, smin=0.05):  # (height scale, z shift m, score factor) applied to the PRIMARY pedestrian boxes; z is box centre
    out = {}
    for f, dets in p.items():
        new = list(dets)
        for x in pred[f]:
            if x['class'] == 'pedestrian' and x['score'] >= smin:
                b = x['box']
                for sh, dz, k in variants:
                    new.append({**x, 'box': [b[0], b[1], b[2] + dz, b[3], b[4], b[5] * sh, b[6]], 'score': x['score'] * k})
        out[f] = new
    return out

SETS = {'current': [], 'h2': [(0.85, 0, 0.1), (1.15, 0, 0.1)], 'z2': [(1, 0.15, 0.1), (1, -0.15, 0.1)],
        'hz4': [(0.85, 0, 0.1), (1.15, 0, 0.1), (1, 0.15, 0.05), (1, -0.15, 0.05)]}
for label, v in SETS.items():
    p = hh(base, v)
    q = {f: [x for x in p[f] if x['class'] == 'pedestrian'] for f in ids}
    out = []
    for h in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: q[f] for f in h}, {f: gt[f] for f in h}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 2))
    print(f'{d.name} {label:8s} pedestrian AP {out} boxes/frame {sum(len(v) for v in q.values()) / len(ids):.0f}', flush=True)
