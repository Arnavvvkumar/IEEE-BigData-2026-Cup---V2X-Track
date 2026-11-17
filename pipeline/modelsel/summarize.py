"""Table of real-dev mAP (all 50 / selection 25 / confirmation 25) for every scored checkpoint under modelsel/real."""
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
C = ['vehicle', 'pedestrian', 'truck']
rows = []
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('real')
for d in sorted(root.iterdir()):
    p = d / 'predictions.json'
    if not p.exists():
        continue
    pred = json.loads(p.read_text())
    row = {'run': d.name}
    for name, ids in (('all', list(gt)), ('sel', split['selection_ids']), ('conf', split['confirmation_ids'])):
        m = ev.evaluate_3d_detection({f: pred.get(f, []) for f in ids}, {f: gt[f] for f in ids}, C, iou_thresholds=(0.5,), iou_type='3d')
        row[name] = round(100 * m['mAP_0.5_R40'], 2)
        if name == 'all':
            for c in C:
                row[c] = round(100 * m['per_class'][c]['AP_0.5_R40'], 2)
    rows.append(row)
rows.sort(key=lambda r: -r['sel'])
print(f"{'run':24s} {'all':>6s} {'sel':>6s} {'conf':>6s} {'veh':>6s} {'ped':>6s} {'truck':>6s}")
for r in rows:
    print(f"{r['run']:24s} {r['all']:6.2f} {r['sel']:6.2f} {r['conf']:6.2f} {r['vehicle']:6.2f} {r['pedestrian']:6.2f} {r['truck']:6.2f}")
(root.parent / 'summary.json').write_text(json.dumps(rows, indent=1))
