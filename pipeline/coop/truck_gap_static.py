import os
﻿"""Truck dev-vs-test gap, step 2: are dev trucks the same parked objects repeated across frames? And what is our truck AP
when each physical object counts once / when only non-repeated trucks count / per truck type?
usage: truck_gap_static.py <set dir> <norule pred name>   (truck rule a 0.5 base 0.1 + dedupe applied, as in the build)"""
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = {f: [g for g in v if g['points'] >= 5] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
idx = json.loads((d / 'index.json').read_text())
ids = sorted(G)
pred = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))
P = {f: [x for x in pred[f] if x['class'] == 'truck'] for f in ids}

# static = a same-type truck GT within 1.0 m (BEV centre) in another frame of the same scenario
by_scen = defaultdict(list)
for f in ids:
    by_scen[idx[f]['scenario']].append(f)
for f in ids:
    for g in G[f]:
        if g['class'] != 'truck':
            continue
        others = [h for o in by_scen[idx[f]['scenario']] if o != f for h in G[o] if h['class'] == 'truck' and h['type'] == g['type']]
        g['static'] = any(np.hypot(h['box'][0] - g['box'][0], h['box'][1] - g['box'][1]) < 1.0 for h in others)
tr = [g for f in ids for g in G[f] if g['class'] == 'truck']
print(f'{d.name}: {len(ids)} frames, {len(by_scen)} scenarios, frames/scenario {Counter(len(v) for v in by_scen.values())}')
print('truck GT', len(tr), '| static (repeats in same scenario):', sum(g['static'] for g in tr),
      '| by type static/all:', {t: (sum(g['static'] for g in tr if g['type'] == t), sum(1 for g in tr if g['type'] == t)) for t in sorted({g['type'] for g in tr})})
xs = np.array([g['box'][:2] for g in tr if g['static']])
if len(xs):
    # distinct physical static objects across the whole set (cluster centres within 1.5 m, any scenario)
    cl = []
    for p in xs:
        if not any(np.hypot(*(p - c)) < 1.5 for c in cl):
            cl.append(p)
    print('distinct static truck positions across all scenarios:', len(cl))


def ap(pr, gt, fr):
    m = ev.evaluate_3d_detection({f: pr[f] for f in fr}, {f: gt[f] for f in fr}, ['truck'], iou_thresholds=(0.5,), iou_type='3d')
    c = m['per_class']['truck']
    return round(100 * c['AP_0.5_R40'], 1), c['n_gt']


def ignore(pr, gt_keep, gt_drop):
    """keep GT in gt_keep; drop predictions whose BEV centre lies inside a dropped GT box grown by 1 m (do-not-care region)"""
    out = {}
    for f in ids:
        dr = gt_drop[f]
        keep = []
        for x in pr[f]:
            hit = False
            for g in dr:
                b = g['box']; q = np.array(x['box'][:2]) - b[:2]
                cs, sn = np.cos(-b[6]), np.sin(-b[6])
                if abs(q[0] * cs - q[1] * sn) <= b[3] / 2 + 1 and abs(q[0] * sn + q[1] * cs) <= b[4] / 2 + 1:
                    hit = True; break
            if not hit:
                keep.append(x)
        out[f] = keep
    return out, gt_keep


T = lambda f, cond: [{'class': 'truck', 'box': g['box']} for g in G[f] if g['class'] == 'truck' and cond(g)]
allg = {f: T(f, lambda g: True) for f in ids}
print('\nAP truck (IoU 0.5 R40)                        AP   n_gt')
print('official-like (all truck GT, all frames)   ', ap(P, allg, ids))
one = [v[len(v) // 2] for v in by_scen.values()]
print('one frame per scenario                     ', ap(P, allg, one))
for lab, cond in (('non-static trucks only (static=ignore)', lambda g: not g['static']),
                  ('static trucks only (non-static=ignore)', lambda g: g['static'])):
    pr, gk = ignore(P, {f: T(f, cond) for f in ids}, {f: [g for g in G[f] if g['class'] == 'truck' and not cond(g)] for f in ids})
    print(f'{lab:43s}', ap(pr, gk, ids))
for t in ('TrashCan', 'Bus', 'Truck', 'Van'):
    pr, gk = ignore(P, {f: T(f, lambda g, t=t: g['type'] == t) for f in ids}, {f: [g for g in G[f] if g['class'] == 'truck' and g['type'] != t] for f in ids})
    print(f'type {t:38s}', ap(pr, gk, ids))
# recall by type at IoU 0.5 for any truck prediction with score >= 0.1
print('\nrecall@0.5 (truck preds score>=0.1) by type and static flag:')
rec = defaultdict(lambda: [0, 0])
for f in ids:
    pb = [x['box'] for x in P[f] if x['score'] >= 0.1]
    for g in G[f]:
        if g['class'] != 'truck':
            continue
        ok = any(ev._iou_3d(np.array(g['box']), np.array(b)) >= 0.5 for b in pb)
        k = (g['type'], 'static' if g['static'] else 'moving')
        rec[k][0] += bool(ok); rec[k][1] += 1
print({f'{a}/{b}': f'{v[0]}/{v[1]}' for (a, b), v in sorted(rec.items())})
