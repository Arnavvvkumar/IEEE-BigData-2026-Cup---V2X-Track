"""Compare truck support and anchor coverage on synthetic training and screened dev."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'codex_review'), str(HERE.parent / 'codex_truck')]
from audit import DATA
from replay import screened


def report(name, boxes, counts):
    dims = np.asarray(boxes)[:, 3:6]
    anchor = np.array([8.468, 2.9039])
    intersection = np.minimum(dims[:, :2], anchor).prod(axis=1)
    iou = intersection / (dims[:, :2].prod(axis=1) + anchor.prod() - intersection)
    print(json.dumps({'group': name, 'boxes': len(boxes), 'q10_q50_q90_lwh': np.quantile(dims, [.1, .5, .9], axis=0).round(3).tolist(),
                      'q10_q50_q90_support': np.quantile(counts, [.1, .5, .9]).round(1).tolist(),
                      'centered_BEV_anchor_iou_below_050': float(np.mean(iou < .5)),
                      'fraction_support_at_most_15': float(np.mean(np.asarray(counts) <= 15))}), flush=True)


def synthetic(name):
    root = DATA / name
    files = sorted((root / 'labels').glob('*.txt'))
    rng = np.random.default_rng(1234)
    boxes, counts = [], []
    for index in sorted(rng.choice(len(files), 192, replace=False)):
        path = files[index]
        frame = [np.array(parts[:7], dtype=float) for line in path.read_text().splitlines()
                 if (parts := line.split()) and parts[-1] == 'truck']
        if not frame:
            continue
        cloud = np.load(root / 'points' / f'{path.stem}.npy')[:, :3]
        tree = cKDTree(cloud[:, :2])
        for box in frame:
            x, y, z, l, w, h, yaw = box
            nn = tree.query_ball_point([x, y], np.hypot(l, w) / 2)
            q = cloud[nn] - [x, y, z]
            c, s = np.cos(yaw), np.sin(yaw)
            mask = ((abs(c*q[:, 0] + s*q[:, 1]) <= l / 2)
                    & (abs(-s*q[:, 0] + c*q[:, 1]) <= w / 2)
                    & (abs(q[:, 2]) <= h / 2))
            boxes.append(box)
            counts.append(int(mask.sum()))
    report(name, boxes, counts)


if __name__ == '__main__':
    screened()
    records = json.loads((HERE.parent / 'coop/dev350/gt_coop.json').read_text())
    labels = [g for rows in records.values() for g in rows if g['class'] == 'truck' and g['points'] >= 5]
    for subtype in sorted({g['type'] for g in labels}):
        group = [g for g in labels if g['type'] == subtype]
        report(f'dev350/{subtype}', [g['box'] for g in group], [g['points'] for g in group])
    for name in ('claude_canon_all_v1', 'claude_twin_st_v1', 'claude_twin_pv_v1'):
        synthetic(name)
