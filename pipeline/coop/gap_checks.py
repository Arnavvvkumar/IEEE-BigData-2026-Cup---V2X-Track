"""Two follow-up checks.
 1) Scan density: points per scan (total, zeros, non-zero, non-ground) for test, dev350, and a sample of synthetic training frames.
 2) The fixed 'TrashCan' object (labelled truck in the coop labels): where is it, do the TEST truck boxes sit on it, and what is
    dev truck AP under each assumption about the hidden test labels (can counted as truck / can not labelled),
    with our can boxes kept or removed.
usage: gap_checks.py <dev set dir> <norule pred name>"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
F = W / 'experiments/claude_carla_overnight/final'
sys.path.insert(0, str(F))
import calib  # noqa: E402
import fuse_test as FT  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d, name = Path(sys.argv[1]), sys.argv[2]


def dens(clouds, label):
    r = []
    for p in clouds:
        p = np.asarray(p)[:, :3]
        z = (np.abs(p).sum(1) == 0)
        r.append((len(p), z.sum(), (~z).sum(), len(calib.nonground(p))))
    r = np.median(np.array(r), 0).astype(int)
    print(f'  {label:28s} points {r[0]:6d}  zeros {r[1]:6d}  non-zero {r[2]:6d}  non-ground {r[3]:6d}')


print('--- scan density (medians)')
tc = FT.load(W / 'starting_kit/track_b/detection_test_frames')
dens(list(tc.values()), 'test (50 released)')
dc = FT.load(d / 'scans')
dens(list(dc.values())[:60], f'{d.name} (validation)')
D = Path.home() / 'projects/bdc2026-track-b/data'
for s in ('claude_twin_v1', 'claude_canon_all_v1', 'claude_twin_pv_v1'):
    fs = sorted((D / s / 'points').glob('*.npy'))[::max(1, len(list((D / s / 'points').glob('*.npy'))) // 40)][:40]
    dens([np.load(f) for f in fs], f'train {s}')

print('--- trash can')
G = json.loads((d / 'gt_coop.json').read_text())
cans = np.array([g['box'] for v in G.values() for g in v if g.get('type') == 'TrashCan'])
cen = np.median(cans, 0)
print('  can labels', len(cans), 'in', len(G), 'frames | median box', cen.round(2).tolist(), '| range', round(float(np.hypot(cen[0], cen[1])), 1), 'm')
RULE = {**calib.TRUCK, 'a': 0.5, 'base': 0.1}
dev = calib.truck_dedupe(calib.truck_rule(json.loads((d / f'pred_{name}.json').read_text()), **RULE))
test = json.loads((F / 'candidate_v5/predictions.json').read_text())
oncan = lambda x: np.hypot(x['box'][0] - cen[0], x['box'][1] - cen[1]) < 1.5
for lab, p in (('test', test), ('dev', dev)):
    t = [x for f in p for x in p[f] if x['class'] == 'truck']
    c = [x for x in t if oncan(x)]
    top = [max((x['score'] for x in p[f] if x['class'] == 'truck' and oncan(x)), default=0) for f in p]
    print(f'  {lab}: truck boxes >=0.3: {sum(x["score"] >= .3 for x in t)} of which on the can spot {sum(x["score"] >= .3 for x in c)} | frames with a can-spot truck box >=0.3: {sum(s >= .3 for s in top)}/{len(p)} | median top can-spot score {np.median(top):.2f}')
    n = [len(q) for q in ([calib.nonground(np.asarray(cl)[:, :3]) for cl in list((tc if lab == "test" else dc).values())[:50]])]
    pts = []
    for cl in list((tc if lab == 'test' else dc).values())[:50]:
        q = calib.nonground(np.asarray(cl)[:, :3])
        pts.append(int((np.hypot(q[:, 0] - cen[0], q[:, 1] - cen[1]) < 1.0).sum()))
    print(f'       non-ground points within 1 m of the can spot: median {int(np.median(pts))}')
ids = sorted(G)
gt_all = {f: [{'class': 'truck', 'box': g['box']} for g in G[f] if g['class'] == 'truck' and g['points'] >= 5] for f in ids}
gt_gen = {f: [{'class': 'truck', 'box': g['box']} for g in G[f] if g['class'] == 'truck' and g['points'] >= 5 and g.get('type') != 'TrashCan'] for f in ids}
keep = {f: [x for x in dev[f] if x['class'] == 'truck'] for f in ids}
drop = {f: [x for x in keep[f] if not oncan(x)] for f in ids}
ap = lambda p, g: round(100 * ev.evaluate_3d_detection(p, g, ['truck'], iou_thresholds=(0.5,), iou_type='3d')['per_class']['truck']['AP_0.5_R40'], 1)
print('  dev truck AP              our can boxes KEPT   our can boxes REMOVED')
print(f'  if the can IS a truck label      {ap(keep, gt_all):6.1f}              {ap(drop, gt_all):6.1f}')
print(f'  if the can is NOT labelled       {ap(keep, gt_gen):6.1f}              {ap(drop, gt_gen):6.1f}')
print('CHECKS_DONE')
