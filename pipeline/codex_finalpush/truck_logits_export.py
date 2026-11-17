"""Export original or truck-only class scores from a fixed SECOND checkpoint."""
import os
import runpy
import sys
from pathlib import Path

from importlib import import_module

exp = Path(__file__).resolve().parent.parent
args = sys.argv[1:]
repo = Path(args[args.index('--repo') + 1]).resolve()
sys.path[:0] = [str(repo), str(exp / 'final'), str(exp / 'codex_review')]
from per_scan_export import fit_independently
from pcdet.models.detectors.detector3d_template import Detector3DTemplate

batched = import_module('batched_forward')
batched.fit_planes = fit_independently
stock = Detector3DTemplate.post_processing


def truck_scores(self, batch_dict):
    if batch_dict.get('has_class_labels', False):
        raise RuntimeError('Only the three-class SECOND head uses this diagnostic')
    raw = batch_dict['batch_cls_preds']
    if not hasattr(raw, 'shape') or raw.shape[-1] != 3 or self.class_names[2] != 'truck':
        raise RuntimeError('Unexpected class ordering')
    masked = raw.clone()
    masked[..., :2] = 0. if batch_dict['cls_preds_normalized'] else float('-inf')
    return stock(self, {**batch_dict, 'batch_cls_preds': masked})


if os.environ.get('TRUCK_LOGIT_MODE') == 'retain':
    Detector3DTemplate.post_processing = truck_scores
os.environ['PLANE'] = 'scan'
sys.argv = [str(exp / 'final/infer_test.py'), *args, '--batch', '1']
runpy.run_path(sys.argv[0], run_name='__main__')
