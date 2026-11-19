"""Codex idea 1: which part of the v17 five-point rule helped? The rule (a) lowers the primary's score x0.25 and (b) pushes some parents below
the hedge-eligibility cut (score >= 0.05), which both removes their hedge family and lowers variant scores (k x parent score).
Pre-registered variants (n = non-ground points in the box grown 0.1 m, flagged = n < 5), all on cached v16-base predictions, hedged, 3 decimals:
  REF  v17 as deployed.
  A    score-only: flagged primary x0.25, but every parent that was eligible BEFORE the rule keeps its hedge family (variant score = k x reduced parent score).
  B    hedge-removal-only: primary scores unchanged; flagged parents get no hedge family.
  C    flagged primary x0.25, hedge family for every originally-eligible parent at k x ORIGINAL score (variants of flagged boxes not penalised).
  NONE no rule (v16 reference).
usage: ped_minpts_decomp.py <set dir> <ped-only norule pred name> [workers]"""
import os
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]
G = json.loads((d / 'gt_coop.json').read_text())
ids = sorted(G)
gtd = {f: [{'class': 'pedestrian', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
clouds = {f: np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
pp = json.loads((d / f'pred_{name}.json').read_text())
O = {f: [x for x in pp[f] if x['class'] == 'pedestrian'] for f in ids}
R = calib.ped_min_points(O, clouds)
FLAG = {f: [r['score'] != o['score'] for o, r in zip(O[f], R[f])] for f in ids}
H = calib.PED_HEDGE


def family(x, score):
    b = x['box']
    return [{**x, 'box': [b[0] + dx, b[1] + dy, b[2], b[3] * sl, b[4] * sw, b[5], b[6]], 'score': score * k} for sl, sw, dx, dy, k in H]


def build(mode):
    out = {}
    for f in ids:
        new = []
        for o, r, fl in zip(O[f], R[f], FLAG[f]):
            elig = o['score'] >= 0.05
            if mode == 'REF':
                new.append(r)
                if r['score'] >= 0.05: new += family(r, r['score'])
            elif mode == 'NONE':
                new.append(o)
                if elig: new += family(o, o['score'])
            elif mode == 'A':
                new.append(r)
                if elig: new += family(r, r['score'])
            elif mode == 'B':
                new.append(o)
                if elig and not fl: new += family(o, o['score'])
            elif mode == 'C':
                new.append(r)
                if elig: new += family(r, o['score'] if fl else r['score'])
        out[f] = new
    return out


def ap(p):
    out = []
    for hh in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: p[f] for f in hh}, {f: gtd[f] for f in hh}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 3))
    return out


def run(mode):
    p = build(mode)
    return f'{d.name} {mode:5s} boxes/frame {sum(map(len, p.values())) / len(ids):.0f} | ped AP {ap(p)}'


if __name__ == '__main__':
    nfl = sum(map(sum, FLAG.values())); ne = sum(1 for f in ids for o in O[f] if o['score'] >= 0.05); nfe = sum(1 for f in ids for o, fl in zip(O[f], FLAG[f]) if fl and o['score'] >= 0.05)
    print(f'{d.name} parents {sum(map(len, O.values()))} | flagged {nfl} | eligible before rule {ne} | flagged AND eligible {nfe}', flush=True)
    from multiprocessing import get_context
    with get_context('fork').Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 3) as pool:
        for line in pool.imap(run, ['NONE', 'REF', 'A', 'B', 'C']):
            print(line, flush=True)
    print('DECOMP_DONE')
