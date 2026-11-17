"""Score the current ensemble (fuse_cw_d + pedestrian static down-weight + calibration) on a staged validation set and break
the result down by raw object type. Labels: cooperative, >= 5 sensor points (closest proxy to the organizers' labels).
usage: analyze_dev.py <set dir (scans/, gt_coop.json, allpred/)>"""
import importlib.util
import json
import sys
from collections import defaultdict
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
d = Path(sys.argv[1])
gt_all = json.loads((d / 'gt_coop.json').read_text())
gt = {f: [g for g in v if g['points'] >= 5] for f, v in gt_all.items()}
ids = list(gt)
cache = d / 'pred_cw_d_calibrated.json'
if cache.exists():
    pred = json.loads(cache.read_text())
else:
    clouds = FT.load(d / 'scans')
    A = d / 'allpred'
    specs = [('vnx010_all_s1234_ep10', 'pedestrian,vehicle'), ('vnx010_all_rs2_amp_ep12', 'pedestrian'), ('canon_all_s1234_ep12', 'vehicle,truck'),
             ('second_twin_s1_ep12', 'vehicle,truck'), ('canon_relabel_s1234_ep12', 'vehicle,truck')]
    passes = []
    for name, keep in specs:
        data = json.loads((A / f'{name}_allpred.json').read_text())
        kc = set(keep.split(','))
        passes += [{f: [x for x in p[f] if x['class'] in kc] for f in p} for p in data.values()]
    fz = FT.fuse(passes, ids)
    (d / 'pred_cw_d_fused.json').write_text(json.dumps(fz))
    st, info = FT.static_down(fz, clouds)
    pred = calib.calibrate(st, clouds)
    cache.write_text(json.dumps(pred))
    print('static info', info)


def ap(p, g, classes=C):
    m = ev.evaluate_3d_detection(p, {f: [{'class': x['class'], 'box': x['box']} for x in v] for f, v in g.items()}, classes, iou_thresholds=(0.5,), iou_type='3d')
    return {c: round(100 * m['per_class'][c]['AP_0.5_R40'], 1) for c in classes}


print('frames', len(ids))
print('AP, coop >=5 pts labels          ', ap(pred, gt))
print('AP, sensor-only labels           ', ap(pred, {f: [g for g in v if g['agent'] == '-1'] for f, v in gt_all.items()}))
print('truck AP, TrashCan NOT labelled  ', ap(pred, {f: [g for g in v if g['type'] != 'TrashCan'] for f, v in gt.items()}, ['truck']))
print('truck AP, only TrashCan labelled ', ap(pred, {f: [g for g in v if g['type'] == 'TrashCan' or g['class'] != 'truck'] for f, v in gt.items()}, ['truck']))
print('real test                         {vehicle: 63.8, pedestrian: 31.3, truck: 52.9}')
rows = defaultdict(list)
for f in ids:
    P = [(np.array(x['box']), x['score'], x['class']) for x in pred[f] if x['score'] >= 0.05]
    for g in gt[f]:
        b = np.array(g['box'])
        near = [(pb, s, c) for pb, s, c in P if np.hypot(pb[0] - b[0], pb[1] - b[1]) < max(b[3], 2.0)]
        same = [(ev._iou_3d(pb, b), s) for pb, s, c in near if c == g['class']]
        other = [(ev._iou_3d(pb, b), s, c) for pb, s, c in near if c != g['class']]
        bs = max(same, default=(0.0, 0.0))
        bo = max(other, default=(0.0, 0.0, ''))
        if bs[0] >= 0.5:
            kind = 'hit'
        elif bo[0] >= 0.5:
            kind = 'wrong class'
        elif bs[0] >= 0.25:
            kind = 'loose box'
        else:
            kind = 'missed'
        rows[(g['class'], g['type'])].append((kind, bs[1] if kind == 'hit' else bo[1] if kind == 'wrong class' else 0.0, np.hypot(b[0], b[1]), g['points']))
print(f"{'class/type':22s}    n   hit  wrong-class  loose-box  missed | median score of hits | missed: median range, points")
for (c, t), r in sorted(rows.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
    k = [x[0] for x in r]
    hits = [x[1] for x in r if x[0] == 'hit']
    miss = [x for x in r if x[0] == 'missed']
    print(f"{c[:4] + '/' + t:22s} {len(r):4d}  {k.count('hit') / len(r):4.2f}     {k.count('wrong class') / len(r):4.2f}       {k.count('loose box') / len(r):4.2f}      {k.count('missed') / len(r):4.2f}  |        {np.median(hits) if hits else 0:4.2f}          |"
          + (f" {np.median([x[2] for x in miss]):5.1f} m, {int(np.median([x[3] for x in miss]))} pts" if miss else ''))
