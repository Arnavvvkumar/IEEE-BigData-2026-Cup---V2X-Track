"""Levelled multi-height inference on a folder of scans, WITHOUT labels (used for the 50 released test scans).
Same procedure as erroranalysis/detilt_eval.py --plane median --save-all, minus the scoring:
ground plane fitted from the scans' own points (median over the folder; fixed sensor), scan levelled to each target ground
height, detector run, boxes mapped back to the sensor frame. Writes {ground: {frame_id: [detections]}}.
usage: infer_test.py --repo R --cfg C --data D --checkpoint K --scans DIR --out OUT.json --ground g1 g2 ..."""
import argparse
import json
import sys
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

p = argparse.ArgumentParser()
for name in ('repo', 'cfg', 'data', 'checkpoint', 'scans', 'out'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--ground', type=float, nargs='+', required=True)
p.add_argument('--batch', type=int, default=4)
p.add_argument('--workers', type=int, default=8)
p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
sys.path.insert(0, str(a.repo))
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / 'erroranalysis'))
from groundplane import fit_ground, level_transform  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from batched_forward import fit_planes, run_frames  # noqa: E402

files = sorted(a.scans.glob('*.bin'))
if a.limit:
    files = files[:a.limit]
ids = [f.stem for f in files]
clouds = {f.stem: np.fromfile(f, dtype='<f4').reshape(-1, 3) for f in files}
planes = fit_planes(fit_ground, clouds, ids, a.workers)
med = np.median(planes, 0)
print('scans', len(ids), 'plane median abc', med.round(4).tolist(), 'std', planes.std(0).round(4).tolist(), flush=True)
# PLANE=scan: level every scan with ITS OWN fitted plane (no information from other scans); default = folder median.
import os  # noqa: E402
PER_SCAN = os.environ.get('PLANE', 'median') == 'scan'
plane_of = {f: (planes[i] if PER_SCAN else med) for i, f in enumerate(ids)}
print('plane mode', 'per-scan' if PER_SCAN else 'folder median', flush=True)

cfg_from_yaml_file(str(a.cfg), cfg)
cfg.DATA_CONFIG.DATA_PATH = str(a.data)
logger = common_utils.create_logger()
dataset, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                 logger=logger, training=False)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
model.load_params_from_file(filename=str(a.checkpoint), logger=logger, to_cpu=True)
model.cuda().eval()
torch.set_num_threads(1)
C = list(cfg.CLASS_NAMES)
ALL = {}
print('batch', a.batch, 'workers', a.workers, flush=True)
packer = ThreadPoolExecutor(max_workers=2)
for g in a.ground:
    RT = {f: level_transform(plane_of[f], g) for f in ids}
    pred = {}
    pending = []

    def decode(copied, _RT, dest):
        for f, b, s, lab in copied:
            _R, _t = _RT[f]
            if len(b):
                h = np.c_[np.cos(b[:, 6]), np.sin(b[:, 6]), np.zeros(len(b))] @ _R
                b[:, :3] = (b[:, :3] - _t) @ _R
                b[:, 6] = np.arctan2(h[:, 1], h[:, 0])
            boxes, scores, labels = b.tolist(), s.tolist(), lab.tolist()
            dest[f] = [{'box': boxes[i], 'score': scores[i], 'class': C[labels[i] - 1]} for i in range(len(boxes))]

    def take(frame_ids, pred_dicts, _RT=RT):
        copied = []
        for f, out in zip(frame_ids, pred_dicts):
            copied.append((f,
                           out['pred_boxes'].detach().cpu().numpy().astype(np.float64, copy=False),
                           out['pred_scores'].detach().cpu().numpy().copy(),
                           out['pred_labels'].detach().cpu().numpy().copy()))
        pending.append(packer.submit(decode, copied, _RT, pred))

    items = [(f, (clouds[f] @ RT[f][0].T + RT[f][1]).astype(np.float32)) for f in ids]
    with torch.inference_mode():
        run_frames(dataset, model, items, a.batch, a.workers, take)
    for job in pending:
        job.result()
    ALL[str(g)] = pred
    print('ground', g, 'detections', sum(len(v) for v in pred.values()), 'gpu_mb', int(torch.cuda.memory_reserved() / 1e6), flush=True)
packer.shutdown(wait=True)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(ALL))
