"""Score an ensemble spec on a staged validation set: fusion -> pedestrian static down-weight -> calibration -> truck size rule.
Labels: cooperative, >= 5 sensor points. Reports AP all / half A / half B per class, and truck AP on genuine trucks.
usage: eval150.py <set dir> <name> <tag_epN>:<classes> [...]   (classes comma-separated; add 'norule' as last arg to skip the truck rule)"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/final'))
import calib  # noqa: E402
import fuse_test as FT  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
C = ['vehicle', 'pedestrian', 'truck']
d, name = Path(sys.argv[1]), sys.argv[2]
args = sys.argv[3:]
dedupe = args[-1] == 'dedupe'  # optional last arg: remove duplicate truck boxes after the truck rule
if dedupe:
    args = args[:-1]
norule = args[-1] == 'norule'
if norule:
    args = args[:-1]
gt = {f: [g for g in v if g['points'] >= 5] for f, v in json.loads((d / 'gt_coop.json').read_text()).items()}
ids = sorted(gt)
halves = [ids, ids[0::2], ids[1::2]]
clouds = FT.load(d / 'scans')
passes, groups = [], []
for a in args:
    tag, keep = a.split(':')
    data = json.loads((d / 'allpred' / f'{tag}_allpred.json').read_text())
    kc = set(keep.split(','))
    passes += [{f: [x for x in p[f] if x['class'] in kc] for f in p} for p in data.values()]
    groups += [tag.replace('_tta', '')] * len(data)   # checkpoint name per pass (TTA grouped with its original)
FT.PASS_GROUP = groups
fz = FT.fuse(passes, ids)
st, _ = FT.static_down(fz, clouds)
pred = calib.calibrate(st, clouds)
if not norule:
    pred = calib.truck_rule(pred, **calib.TRUCK)
if dedupe:
    pred = calib.truck_dedupe(pred)
(d / f'pred_{name}.json').write_text(json.dumps(pred))


def ap(p, g, cls):
    out = []
    for h in halves:
        m = ev.evaluate_3d_detection({f: p[f] for f in h}, {f: [{'class': x['class'], 'box': x['box']} for x in g[f]] for f in h}, [cls], iou_thresholds=(0.5,), iou_type='3d')
        out.append(round(100 * m['per_class'][cls]['AP_0.5_R40'], 1))
    return out


res = {c: ap(pred, gt, c) for c in C}
cans = {f: [np.array(g['box']) for g in gt[f] if g['type'] == 'TrashCan'] for f in ids}
pg = {f: [x for x in pred[f] if not (x['class'] == 'truck' and any(ev._iou_3d(np.array(x['box']), c) >= 0.3 for c in cans[f]))] for f in ids}
res['genuine_truck'] = ap(pg, {f: [g for g in gt[f] if g['type'] != 'TrashCan'] for f in ids}, 'truck')
res['mAP'] = [round(sum(res[c][i] for c in C) / 3, 2) for i in range(3)]
print(name, 'passes', len(passes), json.dumps(res), flush=True)
with (d / 'eval_summary.txt').open('a') as fh:
    fh.write(f"{name} | {' '.join(args)}{' norule' if norule else ''}{' dedupe' if dedupe else ''} | {json.dumps(res)}\n")
