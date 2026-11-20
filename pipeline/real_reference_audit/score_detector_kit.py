"""Score the pinned baseline on public V2X-Real development scenes."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'V2X-Real'))
from opencood.data_utils import SUPER_CLASS_MAP
from opencood.utils.transformation_utils import x1_to_x2


def convert_truth(frame):
    if frame.get('infra') is not True:
        raise ValueError('Expected infrastructure labels')
    classes = {subclass: group for group, names in SUPER_CLASS_MAP.items() for subclass in names}
    truth = []
    for obj in frame['vehicles'].values():
        category = classes.get(obj['obj_type'])
        if category is None:
            continue
        pose = [*(np.asarray(obj['location']) + obj['center']), *obj['angle']]
        transform = x1_to_x2(pose, frame['lidar_pose'])
        center = transform[:3, 3]
        if (center < [-40, -40, -8]).any() or (center > [40, 40, 2]).any():
            continue
        size = 2 * np.asarray(obj['extent'])
        yaw = np.arctan2(transform[1, 0], transform[0, 0])
        box = np.asarray([*center, *size, yaw])
        if not np.isfinite(box).all() or (size <= 0).any():
            raise ValueError(f'Invalid real box: {box}')
        truth.append({'class': category, 'box': box.tolist()})
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kit', type=Path)
    args = parser.parse_args()
    index = json.loads((args.kit / 'source_index.json').read_text())
    truth = {name: convert_truth(yaml.safe_load(Path(row['label']).read_text()))
             for name, row in index.items()}
    pp = json.loads((args.kit / 'PP.json').read_text())
    vn = json.loads((args.kit / 'VN.json').read_text())
    if not (pp.keys() == vn.keys() == truth.keys()):
        raise ValueError('Prediction and label frame IDs disagree')
    combined = {name: [d for d in vn[name] if d['class'] in ('vehicle', 'truck')] +
                [d for d in pp[name] if d['class'] == 'pedestrian'] for name in truth}
    spec = importlib.util.spec_from_file_location('track_b_detection_eval',
                                                 ROOT / 'starting_kit/track_b/scoring_program/detection_eval.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    result = {'scope': 'public real development, one sensor pose', 'frames': len(index),
              'scenarios': len({row['scenario'] for row in index.values()})}
    for name, predictions in [('PointPillars', pp), ('VoxelNeXt', vn), ('CW1', combined)]:
        result[name] = evaluator.evaluate_3d_detection(predictions, truth, ['vehicle', 'pedestrian', 'truck'],
                                                      iou_thresholds=(.5,), iou_type='3d')
    for name, value in [('ground_truth.json', truth), ('CW1.json', combined), ('local_real_scores.json', result)]:
        with (args.kit / name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
