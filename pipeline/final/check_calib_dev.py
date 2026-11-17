"""Score final/calib.py on the real dev frames (all classes; all / selection / confirmation).
usage: check_calib_dev.py <dev predictions.json> [out predictions.json]"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import calib  # noqa: E402

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
gt = json.loads((KIT / 'ground_truth.json').read_text())
split = json.loads((W / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json').read_text())
ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
C = ['vehicle', 'pedestrian', 'truck']
clouds = {f: np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3) for f in ids}
pred = json.loads(Path(sys.argv[1]).read_text())


def score(p):
    row = {}
    for name, sel in (('all', ids), ('sel', split['selection_ids']), ('conf', split['confirmation_ids'])):
        m = ev.evaluate_3d_detection({f: p[f] for f in sel}, {f: gt[f] for f in sel}, C, iou_thresholds=(0.5,), iou_type='3d')
        row[name] = round(100 * m['mAP_0.5_R40'], 2)
        row.update({f'{name}_{c}': round(100 * m['per_class'][c]['AP_0.5_R40'], 2) for c in C})
    return row


print('before', json.dumps(score(pred)), flush=True)
cal = calib.calibrate(pred, clouds)
print('after ', json.dumps(score(cal)), flush=True)
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(cal))
