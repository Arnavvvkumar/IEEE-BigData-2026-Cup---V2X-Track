"""Official Track B detection metric on the real development frames, overall and on each half of the session-disjoint split.
The selection half informed the box-size convention; the confirmation half is the untouched check."""
import importlib.util
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
gt = json.loads((W / 'experiments/real_reference_audit/real_dev_cw1/ground_truth.json').read_text())
split = json.loads((W / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json').read_text())
res = {}
for path in sys.argv[1:]:
    pred = json.loads(Path(path).read_text())
    row = {}
    for name, ids in (('all50', list(gt)), ('selection25', split['selection_ids']), ('confirmation25', split['confirmation_ids'])):
        m = ev.evaluate_3d_detection({f: pred.get(f, []) for f in ids}, {f: gt[f] for f in ids}, ['vehicle', 'pedestrian', 'truck'],
                                     iou_thresholds=(0.5,), iou_type='3d')
        row[name] = {'mAP': round(m['mAP_0.5_R40'], 4), **{c: round(m['per_class'][c]['AP_0.5_R40'], 4) for c in ('vehicle', 'pedestrian', 'truck')}}
    res[path] = row
    print(Path(path).parent.name, json.dumps(row))
