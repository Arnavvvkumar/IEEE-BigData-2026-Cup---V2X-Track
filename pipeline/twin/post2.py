"""Second post-processing stage on fused predictions (sensor frame). Greedy, chosen on the selection half, checked on confirmation.
 - pedestrian box size fixed to the real median size (box-size statistics from the SELECTION half labels only)
 - box bottom placed on the fitted ground plane plus an offset (pedestrian / vehicle)
usage: post2.py <predictions.json> <out.json>"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/erroranalysis'))
from groundplane import fit_ground, level_transform  # noqa: E402

KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
gt = json.loads((KIT / 'ground_truth.json').read_text())
split = json.loads((W / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json').read_text())
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
C = ['vehicle', 'pedestrian', 'truck']
GZ = -3.0
pred0 = json.loads(Path(sys.argv[1]).read_text())
clouds = {f: np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
planes = np.array([fit_ground(clouds[f], np.random.default_rng(i)) for i, f in enumerate(ids)])
R, t = level_transform(np.median(planes, 0), GZ)
size = {c: np.median(np.array([g['box'][3:6] for f in split['selection_ids'] for g in gt[f] if g['class'] == c]), 0) for c in C}
print('selection-half median sizes', {c: size[c].round(3).tolist() for c in C}, flush=True)


def score(pred):
    row = {}
    for name, sel in (('all', ids), ('sel', split['selection_ids']), ('conf', split['confirmation_ids'])):
        m = ev.evaluate_3d_detection({f: pred[f] for f in sel}, {f: gt[f] for f in sel}, C, iou_thresholds=(0.5,), iou_type='3d')
        row[name] = round(100 * m['mAP_0.5_R40'], 2)
        if name == 'all':
            row.update({c: round(100 * m['per_class'][c]['AP_0.5_R40'], 2) for c in C})
    return row


def apply(pred, cls, dims=None, bottom=None):
    """dims: 'lw' or 'lwh' -> replace with median real size; bottom: offset above fitted ground for the box bottom."""
    out = {}
    for f in ids:
        dets = []
        for d in pred[f]:
            if d['class'] == cls:
                b = list(d['box'])
                if dims == 'lw':
                    b[3:5] = size[cls][:2].tolist()
                elif dims == 'lwh':
                    b[3:6] = size[cls].tolist()
                if bottom is not None:
                    c = R @ np.array(b[:3]) + t
                    c[2] = GZ + bottom + b[5] / 2
                    b[:3] = ((c - t) @ R).tolist()
                d = dict(d, box=b)
            dets.append(d)
        out[f] = dets
    return out


cur, cur_row = pred0, score(pred0)
res = {'base': cur_row}
print('base', json.dumps(cur_row), flush=True)
stages = [
    [('ped_lw', dict(cls='pedestrian', dims='lw')), ('ped_lwh', dict(cls='pedestrian', dims='lwh'))],
    [('ped_bottom0.1', dict(cls='pedestrian', bottom=0.1)), ('ped_bottom0.2', dict(cls='pedestrian', bottom=0.2))],
    [('veh_bottom0.0', dict(cls='vehicle', bottom=0.0)), ('veh_bottom0.1', dict(cls='vehicle', bottom=0.1)),
     ('veh_bottom0.2', dict(cls='vehicle', bottom=0.2))],
]
chosen = []
for stage in stages:
    best = (None, cur_row['sel'], None, None)
    for name, kw in stage:
        pv = apply(cur, **kw)
        row = score(pv)
        res['+'.join(chosen + [name])] = row
        print('+'.join(chosen + [name]), json.dumps(row), flush=True)
        if row['sel'] > best[1]:
            best = (name, row['sel'], pv, row)
    if best[0]:
        chosen.append(best[0])
        cur, cur_row = best[2], best[3]
res['chosen'] = chosen
print('chosen', chosen, json.dumps(cur_row))
out = Path(sys.argv[2])
out.write_text(json.dumps(res, indent=1))
out.with_name(out.stem + '_predictions.json').write_text(json.dumps(cur))
