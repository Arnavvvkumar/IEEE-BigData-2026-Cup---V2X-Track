"""Does a model already detect the static 'TrashCan' container? Nearest truck prediction per selection frame (read-only)."""
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[3]
gt = json.loads((W / 'experiments/real_reference_audit/real_dev_cw1/ground_truth.json').read_text())
split = json.loads((W / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json').read_text())
pred = json.loads((W / sys.argv[1]).read_text())
sys.path.insert(0, str(W / 'starting_kit/track_b/scoring_program'))
import detection_eval as ev  # noqa: E402

iou_fn = getattr(ev, 'boxes_iou3d', None) or getattr(ev, 'iou3d', None)
for f in split['selection_ids']:
    cont = [g for g in gt[f] if g['class'] == 'truck' and abs(g['box'][0] + 3.9) < 1.5 and abs(g['box'][1] - 25.5) < 1.5]
    if not cont:
        continue
    b = np.array(cont[0]['box'])
    ps = [p for p in pred.get(f, []) if np.hypot(p['box'][0] - b[0], p['box'][1] - b[1]) < 4]
    best = max(ps, key=lambda p: p['score'], default=None)
    print(f, 'gt', b.round(2).tolist(), '| nearby preds', [(p['class'], round(p['score'], 2), np.round(p['box'], 2).tolist()) for p in sorted(ps, key=lambda p: -p['score'])[:2]])
