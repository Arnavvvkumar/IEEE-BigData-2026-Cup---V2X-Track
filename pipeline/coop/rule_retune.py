"""Retune the truck size rule + de-duplication on the current ensemble (the rule was set for the v3 ensemble).
Input: calibrated predictions WITHOUT the rule (eval150.py ... norule). Reports vehicle / truck / genuine-truck AP, all / A / B.
usage: rule_retune.py <set dir> <pred name>"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
gt = {f: [g for g in v if g['points'] >= 5 and g['class'] != 'pedestrian'] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
ids = sorted(gt)
halves = [ids, ids[0::2], ids[1::2]]
pred = {f: [x for x in v if x['class'] != 'pedestrian'] for f, v in json.loads((d / f'pred_{name}.json').read_text()).items()}
cans = {f: [np.array(g['box']) for g in gt[f] if g['type'] == 'TrashCan'] for f in ids}
GEN = {f: [g for g in gt[f] if g['type'] != 'TrashCan'] for f in ids}


def ap(p, g, cls):
    out = []
    for h in halves:
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: [{'class': x['class'], 'box': x['box']} for x in g[f]] for f in h}, [cls], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class'][cls]['AP_0.5_R40'], 1))
    return out


def run(job):
    label, kw = job
    kw = dict(kw)
    shrink = kw.pop('shrink', 0.8)
    p = calib.truck_dedupe(calib.truck_rule(pred, **{**calib.TRUCK, **kw}), shrink)
    pg = {f: [x for x in p[f] if not (x['class'] == 'truck' and any(ev._iou_3d(np.array(x['box']), c) >= 0.3 for c in cans[f]))] for f in ids}
    v, t, g = ap(p, gt, 'vehicle'), ap(p, gt, 'truck'), ap(pg, GEN, 'truck')
    return f'{label:34s} vehicle {v} truck {t} genuine {g} | veh+genuine {round(v[0] + g[0], 1)}'


JOBS = [('current (a.6 b.5 base.2)', {}), ('base 0', dict(base=0.0)), ('base 0.1', dict(base=0.1)), ('base 0.3', dict(base=0.3)),
        ('a 0.8', dict(a=0.8)), ('a 1.0', dict(a=1.0)), ('b 0.7', dict(b=0.7)), ('b 0.3', dict(b=0.3)),
        ('a 0.8 b 0.7', dict(a=0.8, b=0.7)), ('H 2.1', dict(H=2.1)), ('L 5.5', dict(L=5.5)), ('K 3', dict(K=3)),
        ('dedupe shrink 1.0', dict(shrink=1.0)), ('dedupe shrink 0.6', dict(shrink=0.6)),
        ('base 0.1 shrink 0.6', dict(base=0.1, shrink=0.6)), ('base 0.05', dict(base=0.05)), ('base 0.15', dict(base=0.15)),
        ('a 0.5', dict(a=0.5)), ('a 0.5 base 0.1', dict(a=0.5, base=0.1)),
        ('lowA a 0.4 base 0.1', dict(a=0.4, base=0.1)), ('lowA a 0.3 base 0.1', dict(a=0.3, base=0.1)),
        ('lowA a 0.2 base 0.1', dict(a=0.2, base=0.1)), ('lowA a 0.3 b 0.3 base 0.1', dict(a=0.3, b=0.3, base=0.1)),
        ('lowA a 0.3 base 0.05', dict(a=0.3, base=0.05)), ('lowA a 0.2 base 0.05', dict(a=0.2, base=0.05)),
        ('lowA a 0.15 base 0.05', dict(a=0.15, base=0.05)), ('lowA a 0.1 base 0.05', dict(a=0.1, base=0.05))]
if len(sys.argv) > 4:  # optional: only the jobs whose label contains one of these comma-separated strings
    JOBS = [j for j in JOBS if any(k in j[0] for k in sys.argv[4].split(','))]
if __name__ == '__main__':
    from multiprocessing import get_context
    with get_context('fork').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 5) as pool:  # forked workers share the loaded data
        for line in pool.imap(run, JOBS):
            print(line, flush=True)
    print('RULE_RETUNE_DONE')
