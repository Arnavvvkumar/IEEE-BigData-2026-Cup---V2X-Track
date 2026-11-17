"""Score uncalibrated vs calibrated predictions against sensor-only labels and cooperative labels (all / >=5 sensor points).
usage: score_variants.py <coop gt json> <uncalibrated dev predictions> <calibrated dev predictions>"""
import importlib.util
import json
import sys
from pathlib import Path

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
C = ['vehicle', 'pedestrian', 'truck']
coop = json.loads(Path(sys.argv[1]).read_text())
old = json.loads((W / 'experiments/real_reference_audit/real_dev_cw1/ground_truth.json').read_text())
sets = {'sensor-only': old,
        'coop all': coop,
        'coop >=5 pts': {f: [g for g in v if g['points'] >= 5] for f, v in coop.items()}}
for pn, path in (('uncalibrated', sys.argv[2]), ('calibrated  ', sys.argv[3])):
    pred = json.loads(Path(path).read_text())
    for sn, g in sets.items():
        m = ev.evaluate_3d_detection(pred, {f: [{'class': x['class'], 'box': x['box']} for x in v] for f, v in g.items()}, C, iou_thresholds=(0.5,), iou_type='3d')
        print(pn, f'{sn:13s}', 'mAP', round(100 * m['mAP_0.5_R40'], 2), {c: round(100 * m['per_class'][c]['AP_0.5_R40'], 2) for c in C})
print('real test (calibrated, submitted)   mAP 49.33 {vehicle: 63.76, pedestrian: 31.3, truck: 52.92}')
