"""Test-time ground levelling on the real dev frames (no retraining, no labels used by the transform).
For each real scan: fit the ground plane from the points alone (constrained RANSAC), rotate the scan so the ground is level,
shift it to a target ground height, run the detector, and map the boxes back to the original sensor frame.
Sweeps the target ground height. Scores all 50 / selection 25 / confirmation 25 with the official evaluator.
--plane points : per-frame plane from points; --plane median : median of the per-frame point planes (static sensor);
--plane none : no levelling (baseline check)."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser()
for name in ('repo', 'cfg', 'data', 'checkpoint', 'out'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--plane', default='median')
p.add_argument('--ground', type=float, nargs='+', default=[-5.2])
p.add_argument('--save-best', action='store_true')
p.add_argument('--save-all', action='store_true')  # also write <out>_allpred.json {ground: predictions} for fusion
a = p.parse_args()
ALL = {}
W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
sys.path.insert(0, str(a.repo))
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from groundplane import fit_ground, level_transform  # noqa: E402

spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
gt = json.loads((KIT / 'ground_truth.json').read_text())
split = json.loads((W / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json').read_text())
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
clouds = {f: np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
planes = {f: fit_ground(clouds[f], np.random.default_rng(i)) for i, f in enumerate(ids)}
med = np.median(np.array(list(planes.values())), 0)
print('point-fit plane median abc', med.round(4).tolist(), 'std', np.array(list(planes.values())).std(0).round(4).tolist(), flush=True)

cfg_from_yaml_file(str(a.cfg), cfg)
cfg.DATA_CONFIG.DATA_PATH = str(a.data)
logger = common_utils.create_logger()
dataset, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                 logger=logger, training=False)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
model.load_params_from_file(filename=str(a.checkpoint), logger=logger, to_cpu=True)
model.cuda().eval()
C = list(cfg.CLASS_NAMES)


def score(pred):
    row = {}
    for name, sel in (('all', ids), ('sel', split['selection_ids']), ('conf', split['confirmation_ids'])):
        m = ev.evaluate_3d_detection({f: pred[f] for f in sel}, {f: gt[f] for f in sel}, C, iou_thresholds=(0.5,), iou_type='3d')
        row[name] = round(100 * m['mAP_0.5_R40'], 2)
        if name == 'all':
            row.update({c: round(100 * m['per_class'][c]['AP_0.5_R40'], 2) for c in C})
    return row


res, best = {}, (None, -1, None)
for g in ([None] if a.plane == 'none' else a.ground):
    pred = {}
    with torch.no_grad():
        for f in ids:
            xyz = clouds[f]
            if a.plane == 'none':
                R, t = np.eye(3), np.zeros(3)
            else:
                R, t = level_transform(planes[f] if a.plane == 'points' else med, g)
            pts = (xyz @ R.T + t).astype(np.float32)
            batch = dataset.collate_batch([dataset.prepare_data({'points': np.pad(pts, ((0, 0), (0, 1))), 'frame_id': f})])
            load_data_to_gpu(batch)
            out = model(batch)[0][0]
            b = out['pred_boxes'].cpu().numpy().astype(np.float64)
            s = out['pred_scores'].cpu().numpy()
            lab = out['pred_labels'].cpu().numpy()
            if len(b):
                h = np.c_[np.cos(b[:, 6]), np.sin(b[:, 6]), np.zeros(len(b))] @ R  # R^T applied to row vectors
                b[:, :3] = (b[:, :3] - t) @ R
                b[:, 6] = np.arctan2(h[:, 1], h[:, 0])
            pred[f] = [{'box': bb.tolist(), 'score': float(ss), 'class': C[int(ll) - 1]} for bb, ss, ll in zip(b, s, lab)]
    row = score(pred)
    res[str(g)] = row
    if a.save_all:
        ALL[str(g)] = pred
    print(a.plane, 'ground', g, json.dumps(row), flush=True)
    if row['sel'] > best[1]:
        best = (g, row['sel'], pred)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({'plane_mode': a.plane, 'point_plane_median_abc': med.tolist(), 'checkpoint': str(a.checkpoint), 'results': res,
                             'best_ground_by_selection': best[0]}, indent=1))
if a.save_all:
    a.out.with_name(a.out.stem + '_allpred.json').write_text(json.dumps(ALL))
if a.save_best:
    a.out.with_name(a.out.stem + '_predictions.json').write_text(json.dumps(best[2]))
