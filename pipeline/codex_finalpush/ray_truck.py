"""Single-scan free-space diagnostic for vehicle-derived truck detections."""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'codex_postmortem'), str(BASE / 'codex_truck')]
from truck_replay import ap, frame_rows, iou, read, screened
from audit import add_postprocess, copy_rule


def ray_counts(box, points, ranges):
    """Intersect measured rays with an inset oriented box above its underbody."""
    center = np.asarray(box[:3])
    half = np.asarray(box[3:6]) / 2
    lower = -half + [.20, .15, .60]
    upper = half - [.20, .15, .20]
    if np.any(lower >= upper):
        return 0, 0
    # Cull rays using a sphere enclosing the full box before intersecting slabs.
    radius = np.linalg.norm(half)
    distance = np.linalg.norm(center)
    projection = points @ center
    eligible = (ranges > max(0., distance - radius)) & (
        projection >= ranges * np.sqrt(max(0., distance**2 - radius**2)))
    depth = ranges[eligible]
    directions = points[eligible] / depth[:, None]
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    local = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    origin = -center @ local
    direction = directions @ local
    parallel = np.abs(direction) < 1e-8
    with np.errstate(divide='ignore', invalid='ignore'):
        a = (lower - origin) / direction
        b = (upper - origin) / direction
    near = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=1)
    far = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=1)
    valid = ~np.any(parallel & ((origin < lower) | (origin > upper)), axis=1)
    valid &= (near > 0) & (far - near >= .30) & (depth >= near - .20)
    through = valid & (depth >= far + .25)
    return int(through.sum()), int(valid.sum())


def contradicted(counts):
    through, observed = counts
    return through >= 5 and through >= .5 * observed


def smoke():
    box = [10, 0, 0, 4, 2, 2, 0]
    points = np.array([[5., 0, 0], [9., 0, 0], [15., 0, 0],
                       [0., 0, 0], [15., 0, -1.5], [15., 4, 0]])
    assert ray_counts(box, points, np.linalg.norm(points, axis=1)) == (1, 2)
    rotated = points[:, [1, 0, 2]].copy()
    rotated[:, 0] *= -1
    rotated_box = [0, 10, 0, 4, 2, 2, np.pi / 2]
    assert ray_counts(rotated_box, rotated, np.linalg.norm(rotated, axis=1)) == (1, 2)


def run(size):
    screened()
    directory = BASE / 'coop' / f'dev{size}'
    truth = {f: [g for g in rows if g['points'] >= 5]
             for f, rows in read(directory / 'gt_coop.json').items()}
    base = copy_rule(read(directory / 'pred_v12p3x3_norule.json'))
    assert len(base) == size and base.keys() == truth.keys()
    groups = {'truck_match': [], 'car_overlap': []}
    for frame in sorted(base):
        points = np.fromfile(directory / 'scans' / f'{frame}.bin', dtype='<f4').reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        depth = np.linalg.norm(points, axis=1)
        copies = [d for d in base[frame] if d['vehicle_copy']]
        truck = iou([d['box'] for d in copies],
                    [g['box'] for g in truth[frame] if g['class'] == 'truck'])
        cars = iou([d['box'] for d in copies],
                   [g['box'] for g in truth[frame] if g['class'] == 'vehicle'])
        for j, detection in enumerate(copies):
            counts = ray_counts(detection['box'], points, depth)
            detection['contradicted'] = contradicted(counts)
            if truck[j].max(initial=0) >= .5:
                groups['truck_match'].append(counts)
            elif cars[j].max(initial=0) >= .5:
                groups['car_overlap'].append(counts)
    control = add_postprocess(base)
    candidate = {f: [{**d, 'score': d['score'] * .5} if d.get('contradicted') else d
                     for d in rows] for f, rows in control.items()}
    print(json.dumps({'set': size, 'groups': {
        key: {'total': len(values), 'contradicted': sum(map(contradicted, values))}
        for key, values in groups.items()},
        'baseline': ap(frame_rows(control, truth, 'truck')),
        'candidate': ap(frame_rows(candidate, truth, 'truck')),
        'baseline_genuine': ap(frame_rows(control, truth, 'truck', genuine=True)),
        'candidate_genuine': ap(frame_rows(candidate, truth, 'truck', genuine=True))}),
        flush=True)


if __name__ == '__main__':
    smoke()
    for arg in sys.argv[1:]:
        run(int(arg))
