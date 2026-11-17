"""Measure separation from the original fixed ray feature, by distance."""
import json
import sys

import numpy as np
from scipy.stats import mannwhitneyu

from ray_truck import BASE, copy_rule, iou, ray_counts, read, screened


def run(size):
    directory = BASE / 'coop' / f'dev{size}'
    truth = read(directory / 'gt_coop.json')
    pred = copy_rule(read(directory / 'pred_v12p3x3_norule.json'))
    assert len(pred) == size and pred.keys() == truth.keys()
    bands = [(0, 20), (20, 35), (35, float('inf'))]
    groups = {(a, b): [[], []] for a, b in bands}
    for frame, rows in pred.items():
        copies = [d['box'] for d in rows if d['vehicle_copy']]
        labels = [g for g in truth[frame] if g['points'] >= 5]
        truck = iou(copies, [g['box'] for g in labels if g['class'] == 'truck']).max(axis=1, initial=0)
        car = iou(copies, [g['box'] for g in labels if g['class'] == 'vehicle']).max(axis=1, initial=0)
        points = np.fromfile(directory / 'scans' / f'{frame}.bin', dtype='<f4').reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        depth = np.linalg.norm(points, axis=1)
        for box, truck_iou, car_iou in zip(copies, truck, car, strict=True):
            if max(truck_iou, car_iou) < .5:
                continue
            through, observed = ray_counts(box, points, depth)
            distance = np.hypot(*box[:2])
            band = next((a, b) for a, b in bands if a <= distance < b)
            groups[band][int(truck_iou >= .5)].append((through, observed))
    for (lo, hi), (cars, trucks) in groups.items():
        def rate(rows):
            return [passed / count for passed, count in rows if count >= 8]

        a, b = rate(cars), rate(trucks)
        auc = mannwhitneyu(a, b).statistic / (len(a) * len(b)) if a and b else None
        print(json.dumps(dict(set=size, band=[lo, hi if np.isfinite(hi) else None],
                              car_total=len(cars), truck_total=len(trucks),
                              car_scored=len(a), truck_scored=len(b), auc=auc)), flush=True)


if __name__ == '__main__':
    screened()
    for arg in sys.argv[1:]:
        run(int(arg))
