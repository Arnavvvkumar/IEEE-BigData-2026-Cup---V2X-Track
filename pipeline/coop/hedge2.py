"""Box hedging, second round, for any class: every box (score >= smin) is also emitted as variants at score * f.
A variant = (length scale, width scale, shift along the box's own length axis in m, sideways shift in m, height scale with the box
bottom kept, f). For pedestrians the shifts are along world x / y (their heading is arbitrary).
Reports AP all / half A / half B (and genuine-truck AP for trucks) and boxes per frame.
usage: hedge2.py <set dir> <norule pred name> <class> [workers]"""
import os
import importlib.util
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d, name, SPEC_KEY = Path(sys.argv[1]), sys.argv[2], sys.argv[3]   # spec key: pedestrian | pedestrian3 | vehicle | truck
CLS = 'pedestrian' if SPEC_KEY.startswith('pedestrian') else SPEC_KEY.rstrip('2')
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
halves = [ids, ids[0::2], ids[1::2]]
gt = {f: [g for g in G[f] if g['points'] >= 5 and g['class'] == CLS] for f in ids}
full = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **{**calib.TRUCK, 'a': 0.5, 'base': 0.1}))
pred = {f: [x for x in full[f] if x['class'] == CLS] for f in ids}
del full
cans = {f: [np.array(g['box']) for g in gt[f] if g.get('type') == 'TrashCan'] for f in ids}

S4 = [(0.8, 0.8), (1.25, 1.25), (1.0, 1.4), (1.4, 1.0)]
SH = lambda r: [(r, 0), (-r, 0), (0, r), (0, -r)]
sz = lambda pairs, f: [(a, b, 0, 0, 1, f) for a, b in pairs]
sh = lambda pairs, f: [(1, 1, a, b, 1, f) for a, b in pairs]
SPECS = {
    'pedestrian': {
        'base': [],
        'sizes4 f.1 (round 1 best)': sz(S4, .1),
        'sizes6 f.1': sz(S4 + [(0.65, 0.65), (1.6, 1.6)], .1),
        'sizes4 f.1 + shifts .15 f.05': sz(S4, .1) + sh(SH(.15), .05),
        'sizes4 f.1 + shifts .25 f.03': sz(S4, .1) + sh(SH(.25), .03),
        'sizes4 f.1 + diag shifts .12 f.05': sz(S4, .1) + sh([(.12, .12), (.12, -.12), (-.12, .12), (-.12, -.12)], .05),
        'sizes4 f.1 + shifts f.05 + size x shift f.02': sz(S4, .1) + sh(SH(.15), .05) + [(a, b, x, y, 1, .02) for a, b in S4 for x, y in SH(.15)],
        'sizes4 f.1 + height .88/1.12 f.05': sz(S4, .1) + [(1, 1, 0, 0, .88, .05), (1, 1, 0, 0, 1.12, .05)],
        'sizes8 graded': sz([(0.8, 0.8), (1.25, 1.25)], .2) + sz([(1.0, 1.4), (1.4, 1.0)], .1) + sz([(0.65, 0.65), (1.6, 1.6), (0.8, 1.2), (1.2, 0.8)], .05),
        'sizes4 f.1 + riders f.03': sz(S4, .1) + sz([(2.0, 0.9), (0.9, 2.0), (1.5, 1.0), (1.0, 1.5)], .03),
    },
    'vehicle': {
        'base': [],
        'shift along +-0.4 f.1': sh([(.4, 0), (-.4, 0)], .1),
        'shift along +-0.4, side +-0.25 f.1': sh([(.4, 0), (-.4, 0), (0, .25), (0, -.25)], .1),
        'length .88/1.12 f.1': sz([(.88, 1), (1.12, 1)], .1),
        'length+width .9/1.1 f.1': sz([(.9, .9), (1.1, 1.1)], .1),
        'height .9/1.12 f.1': [(1, 1, 0, 0, .9, .1), (1, 1, 0, 0, 1.12, .1)],
        'shifts + lengths f.05': sh([(.4, 0), (-.4, 0), (0, .25), (0, -.25)], .05) + sz([(.88, 1), (1.12, 1)], .05),
    },
    'truck': {
        'base': [],
        'length .75/1.3/1.6 f.1': sz([(.75, 1), (1.3, 1), (1.6, 1)], .1),
        'shift along +-1 f.1': sh([(1, 0), (-1, 0)], .1),
        'shift along +-1, +-2 f.1': sh([(1, 0), (-1, 0), (2, 0), (-2, 0)], .1),
        'length + shift f.1': sz([(.75, 1), (1.3, 1), (1.6, 1)], .1) + sh([(1, 0), (-1, 0)], .1),
        'height .85/1.2/1.4 f.1': [(1, 1, 0, 0, .85, .1), (1, 1, 0, 0, 1.2, .1), (1, 1, 0, 0, 1.4, .1)],
        'length + height f.1': sz([(.75, 1), (1.3, 1), (1.6, 1)], .1) + [(1, 1, 0, 0, .85, .1), (1, 1, 0, 0, 1.2, .1), (1, 1, 0, 0, 1.4, .1)],
        'length x height f.05': [(a, 1, 0, 0, h, .05) for a in (.75, 1.3, 1.6) for h in (.85, 1.2, 1.4)],
    },
}
G8 = sz([(0.8, 0.8), (1.25, 1.25)], .2) + sz([(1.0, 1.4), (1.4, 1.0)], .1) + sz([(0.65, 0.65), (1.6, 1.6), (0.8, 1.2), (1.2, 0.8)], .05)
DG = [(.12, .12), (.12, -.12), (-.12, .12), (-.12, -.12)]
SPECS['pedestrian3'] = {   # round 3: build on 'sizes8 graded' (dev150 41.9, dev350 40.8)
    'G8 (round 2 best)': G8,
    'G8 + diag shifts .12 f.05': G8 + sh(DG, .05),
    'G8 + diag shifts .12 f.1': G8 + sh(DG, .1),
    'G8 + axis shifts .15 f.07': G8 + sh(SH(.15), .07),
    'G12: G8 + .9/1.12 f.3 + .72/1.4 f.1': G8 + sz([(0.9, 0.9), (1.12, 1.12)], .3) + sz([(0.72, 0.72), (1.4, 1.4)], .1),
    'G8 steeper grades .3/.15/.07': sz([(0.8, 0.8), (1.25, 1.25)], .3) + sz([(1.0, 1.4), (1.4, 1.0)], .15) + sz([(0.65, 0.65), (1.6, 1.6), (0.8, 1.2), (1.2, 0.8)], .07),
    'G8 flatter grades .12/.08/.05': sz([(0.8, 0.8), (1.25, 1.25)], .12) + sz([(1.0, 1.4), (1.4, 1.0)], .08) + sz([(0.65, 0.65), (1.6, 1.6), (0.8, 1.2), (1.2, 0.8)], .05),
    'G12 + diag shifts f.05': G8 + sz([(0.9, 0.9), (1.12, 1.12)], .3) + sz([(0.72, 0.72), (1.4, 1.4)], .1) + sh(DG, .05),
    'G8 + diag shifts f.05 + (.8/1.25) x diag f.03': G8 + sh(DG, .05) + [(a, a, x, y, 1, .03) for a in (0.8, 1.25) for x, y in DG],
}
LH = lambda ls, hs, f: [(a, 1.0, 0, 0, h, f) for a in ls for h in hs]
SPECS['truck2'] = {   # round 2 for trucks: variants at much higher score fractions, aimed at under-sized boxes of big vehicles
    'base': [],
    'v9 hedge (L x H grid f.05)': [(a, 1.0, 0, 0, h, .05) for a in (.75, 1.3, 1.6) for h in (.85, 1.2, 1.4)],
    'long x1.35 f.3': LH([1.35], [1.0], .3),
    'long x1.35, tall x1.3 f.3': LH([1.35], [1.3], .3),
    'long 1.35/1.6 x tall 1.0/1.3 f.2': LH([1.35, 1.6], [1.0, 1.3], .2),
    'long 1.35/1.6 x tall 1.0/1.3 f.4': LH([1.35, 1.6], [1.0, 1.3], .4),
    'wide too: L1.4 W1.2 H1.3 f.3': [(1.4, 1.2, 0, 0, 1.3, .3)],
    'tall x1.3 only f.3': LH([1.0], [1.3], .3),
    'short x.8 f.3': LH([.8], [1.0], .3),
}
SMIN = {'pedestrian': 0.05, 'vehicle': 0.03, 'truck': 0.02}[CLS]


def make(b, v):
    sl, sw, da, ds, hs, _ = v
    x, y, z, l, w, h, yaw = b
    if CLS == 'pedestrian':
        x, y = x + da, y + ds
    else:
        x, y = x + da * np.cos(yaw) - ds * np.sin(yaw), y + da * np.sin(yaw) + ds * np.cos(yaw)
    nh = h * hs
    return [float(x), float(y), z - h / 2 + nh / 2, l * sl, w * sw, nh, yaw]


def ap(p, g):
    out = []
    for hh in halves:
        m = ev.evaluate_3d_detection({f: p[f] for f in hh}, {f: [{'class': CLS, 'box': x['box']} for x in g[f]] for f in hh}, [CLS], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class'][CLS]['AP_0.5_R40'], 1))
    return out


def run(label):
    vs = SPECS[SPEC_KEY][label]
    new, n = {}, 0
    for f in ids:
        o = list(pred[f])
        for x in pred[f]:
            if x['score'] >= SMIN:
                o += [{**x, 'box': make(x['box'], v), 'score': x['score'] * v[5]} for v in vs]
        new[f] = o
        n += len(o)
    line = f'{label:46s} boxes/frame {n / len(ids):6.0f}  AP {ap(new, gt)}'
    if CLS == 'truck':
        pg = {f: [x for x in new[f] if not any(ev._iou_3d(np.array(x['box']), c) >= 0.3 for c in cans[f])] for f in ids}
        line += f'  genuine {ap(pg, {f: [g for g in gt[f] if g.get("type") != "TrashCan"] for f in ids})}'
    return line


if __name__ == '__main__':
    with get_context('fork').Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 4) as pool:
        for line in pool.imap(run, list(SPECS[SPEC_KEY])):
            print(line, flush=True)
    print('HEDGE2_DONE', CLS)
