"""Codex idea 2: do sparse pedestrians get penalised twice (static down-weight 0.8 x static fraction, then the 5-point rule)?
Needs predictions made with FUSE_KEEP_SF=1 (each pedestrian carries its static fraction 'sf'). Pre-registered variants, applied ONLY to
point-flagged parents (n < 5 points): the static penalty is softened from 0.8 to 0.4 (S1) or removed (S2) before the x0.25 point factor:
  score' = score / (1 - 0.8 sf) * (1 - D2 sf) * 0.25.   REF = v17 as deployed (D2 = 0.8). Everything else fixed. Hedged, 3 decimals.
usage: ped_static_double.py <set dir> <norule pred name with sf>"""
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
gtd = {f: [{'class': 'pedestrian', 'box': g['box']} for g in G[f] if g['points'] >= 5 and g['class'] == 'pedestrian'] for f in ids}
clouds = {f: np.fromfile(d / 'scans' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
pp = json.loads((d / f'pred_{name}.json').read_text())
O = {f: [x for x in pp[f] if x['class'] == 'pedestrian'] for f in ids}
assert all('sf' in x for f in ids for x in O[f][:3]), 'predictions lack sf: run eval150 with FUSE_KEEP_SF=1'
R = calib.ped_min_points(O, clouds)
FLAG = {f: [r['score'] != o['score'] for o, r in zip(O[f], R[f])] for f in ids}
sfs = np.array([o['sf'] for f in ids for o, fl in zip(O[f], FLAG[f]) if fl]); allsf = np.array([o['sf'] for f in ids for o in O[f]])
print(f'{d.name} flagged {len(sfs)} of {len(allsf)} | flagged with sf>0: {(sfs > 0).mean():.3f} (all boxes {(allsf > 0).mean():.3f}) | mean sf of flagged {sfs.mean():.3f}', flush=True)


def variant(d2):
    out = {}
    for f in ids:
        new = []
        for o, r, fl in zip(O[f], R[f], FLAG[f]):
            if fl and d2 != 0.8:
                s = o['score'] / (1 - 0.8 * o['sf']) * (1 - d2 * o['sf']) * 0.25
                r = {**r, 'score': s}
            new.append(r)
        out[f] = new
    return out


def ap(p):
    out = []
    for hh in (ids, ids[0::2], ids[1::2]):
        m = ev.evaluate_3d_detection({f: p[f] for f in hh}, {f: gtd[f] for f in hh}, ['pedestrian'], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class']['pedestrian']['AP_0.5_R40'], 3))
    return out


for lab, d2 in (('REF (D2 0.8)', 0.8), ('S1 (D2 0.4)', 0.4), ('S2 (D2 0)', 0.0)):
    print(f'{d.name} {lab:14s} ped AP {ap(calib.hedge_pedestrians(variant(d2)))}', flush=True)
print('STATIC_DOUBLE_DONE')
