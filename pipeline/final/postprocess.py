"""Post-processing of a fused prediction file, exactly as in build_candidate_v7.sh (the build of the final submission).
calibrate -> truck_rule (a 0.5, base 0.1) -> truck_dedupe -> [truck_min_points] -> truck_boost_hedge -> [ped_min_points] -> hedge_pedestrians
Switches (environment): TRUCK_MINPTS=1, TRUCKPP=1, TRUCKBOOST=1.0, PED_MINPTS=1, HEDGE=1.
usage: postprocess.py <scans dir> <fused.json> <out predictions.json>"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib  # noqa: E402

clouds = {f.stem: np.fromfile(f, dtype='<f4').reshape(-1, 3) for f in sorted(Path(sys.argv[1]).glob('*.bin'))}
pred = calib.calibrate(json.loads(Path(sys.argv[2]).read_text()), clouds)
rule = {**calib.TRUCK, 'a': float(os.environ.get('TRUCK_A', '0.5')), 'base': float(os.environ.get('TRUCK_BASE', '0.1'))}
pred = calib.truck_dedupe(calib.truck_rule(pred, **rule))
if os.environ.get('TRUCK_MINPTS', '0') == '1':
    pred = calib.truck_min_points(pred, clouds)
if os.environ.get('TRUCKPP', '0') == '1':
    pred = calib.truck_boost_hedge(pred, clouds, boost=float(os.environ.get('TRUCKBOOST', '2.0')))
if os.environ.get('PED_MINPTS', '0') == '1':
    pred = calib.ped_min_points(pred, clouds)
if os.environ.get('HEDGE', '1') == '1':
    pred = calib.hedge_pedestrians(pred)
Path(sys.argv[3]).write_text(json.dumps(pred))
print({c: sum(d['class'] == c for v in pred.values() for d in v) for c in ('vehicle', 'pedestrian', 'truck')})
