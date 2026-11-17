"""CPU error analysis on the 50 public real dev frames (evaluation/statistics only; nothing here feeds training data directly).
Usage: real_error_analysis.py <predictions.json> <out.json>
Sections: scene geometry (ground plane, tilt, range mix), GT statistics (points per box, range, heading, counts),
detection outcome per GT (hit / poor box / wrong class / missed) by range and point count, relaxed-metric APs, false positives."""
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
KIT = W / 'experiments/real_reference_audit/real_dev_cw1'
spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
C = ['vehicle', 'pedestrian', 'truck']
gt = json.loads((KIT / 'ground_truth.json').read_text())
pred = json.loads(Path(sys.argv[1]).read_text())


def in_box(pts, b, margin=0.0):
    c, s = np.cos(-b[6]), np.sin(-b[6])
    d = pts[:, :2] - b[:2]
    x, y = d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c
    return (np.abs(x) <= b[3] / 2 + margin) & (np.abs(y) <= b[4] / 2 + margin) & (np.abs(pts[:, 2] - b[2]) <= b[5] / 2 + margin)


def plane_fit(P):
    A = np.c_[P[:, :2], np.ones(len(P))]
    return np.linalg.lstsq(A, P[:, 2], rcond=None)[0]


def ground_plane(pts, rng):
    """RANSAC plane z = ax + by + c on points within 40 m."""
    p = pts[(np.abs(pts[:, 0]) < 40) & (np.abs(pts[:, 1]) < 40)]
    p = p[rng.choice(len(p), min(len(p), 20000), replace=False)]
    best, bi = None, 0
    for _ in range(300):
        s = p[rng.choice(len(p), 3, replace=False)]
        try:
            co = plane_fit(s)
        except np.linalg.LinAlgError:
            continue
        if np.hypot(co[0], co[1]) > 0.5:
            continue
        inl = np.abs(p[:, 0] * co[0] + p[:, 1] * co[1] + co[2] - p[:, 2]) < 0.15
        if inl.sum() > bi:
            bi, best = inl.sum(), inl
    return plane_fit(p[best]), bi / len(p)


rng = np.random.default_rng(0)
out = {}
planes, npts, share10, share20, zq = [], [], [], [], []
gt_rows = []  # per GT: class, range, npts, outcome...
fp = defaultdict(list)
for f, gts in gt.items():
    pts = np.fromfile(KIT / 'detection_test_frames' / f'{f}.bin', dtype='<f4').reshape(-1, 3)
    co, frac = ground_plane(pts, rng)
    planes.append([*co, frac])
    r = np.hypot(pts[:, 0], pts[:, 1])
    m = (np.abs(pts[:, 0]) <= 40) & (np.abs(pts[:, 1]) <= 40) & (pts[:, 2] >= -8) & (pts[:, 2] <= 2)
    npts.append(int(m.sum())); share10.append(float((r[m] < 10).mean())); share20.append(float((r[m] < 20).mean()))
    zq.append(np.percentile(pts[m, 2], [1, 25, 50, 75, 99]).tolist())
    ps = pred.get(f, [])
    PB = np.array([p['box'] for p in ps]).reshape(-1, 7)
    PC = np.array([p['class'] for p in ps])
    PS = np.array([p['score'] for p in ps])
    matched_pred = np.zeros(len(ps), bool)
    for g in gts:
        b = np.array(g['box'])
        n = int(in_box(pts, b).sum())
        row = {'f': f, 'c': g['class'], 'r': float(np.hypot(b[0], b[1])), 'n': n, 'box': b.tolist(),
               'bottom': float(b[2] - b[5] / 2), 'hag': float(b[2] - b[5] / 2 - (co[0] * b[0] + co[1] * b[1] + co[2]))}
        best_same = best_any = bev_same = 0.0
        best_any_c, best_same_i = None, -1
        if len(ps):
            near = np.where(np.hypot(PB[:, 0] - b[0], PB[:, 1] - b[1]) < 8)[0]
            for i in near:
                iou = ev._iou_3d(PB[i], b)
                if iou > best_any:
                    best_any, best_any_c = iou, PC[i]
                if PC[i] == g['class']:
                    if iou > best_same:
                        best_same, best_same_i = iou, int(i)
                    bev_same = max(bev_same, ev._bev_iou(PB[i], b))
                if iou > 0.1:
                    matched_pred[i] = True
        row.update(iou_same=best_same, iou_any=best_any, any_class=best_any_c, bev_same=bev_same)
        if best_same_i >= 0:
            p = PB[best_same_i]
            row.update(d_xy=float(np.hypot(p[0] - b[0], p[1] - b[1])), d_z=float(p[2] - b[2]), d_lwh=(p[3:6] - b[3:6]).tolist(),
                       d_yaw=float(abs((p[6] - b[6] + np.pi / 2) % np.pi - np.pi / 2)), score=float(PS[best_same_i]))
        row['outcome'] = ('hit' if best_same >= 0.5 else 'poor_box' if best_same >= 0.1 else
                          'wrong_class' if best_any >= 0.3 else 'missed')
        gt_rows.append(row)
    for i in np.where(~matched_pred & (PS >= 0.3))[0]:
        fp[PC[i]].append([round(float(v), 1) for v in PB[i][:3]] + [round(float(PS[i]), 2), f])

P = np.array(planes)
tilt = np.degrees(np.arctan(np.hypot(P[:, 0], P[:, 1])))
out['scene'] = {
    'ground_plane_z=ax+by+c_median': np.median(P[:, :3], 0).round(4).tolist(), 'plane_abc_std': P[:, :3].std(0).round(4).tolist(),
    'tilt_deg_median': round(float(np.median(tilt)), 2), 'downhill_direction_deg': round(float(np.degrees(np.arctan2(-np.median(P[:, 1]), -np.median(P[:, 0])))), 1),
    'sensor_height_above_plane_m': round(float(-np.median(P[:, 2]) / np.sqrt(1 + np.median(P[:, 0]) ** 2 + np.median(P[:, 1]) ** 2)), 2),
    'plane_inlier_fraction_median': round(float(np.median(P[:, 3])), 3),
    'points_in_range_median': int(np.median(npts)), 'share_within_10m': round(float(np.median(share10)), 3),
    'share_within_20m': round(float(np.median(share20)), 3), 'z_percentiles_1_25_50_75_99': np.median(zq, 0).round(2).tolist()}

bins_r = [0, 10, 20, 30, 40, 60]
bins_n = [0, 1, 5, 10, 20, 50, 100, 1e9]
out['per_class'] = {}
for c in C:
    R = [g for g in gt_rows if g['c'] == c]
    if not R:
        continue
    d = {'n_gt': len(R), 'per_frame_median': float(np.median(list(Counter(g['f'] for g in R).values()))),
         'points_per_box_p10_p50_p90': np.percentile([g['n'] for g in R], [10, 50, 90]).round(1).tolist(),
         'range_p10_p50_p90': np.percentile([g['r'] for g in R], [10, 50, 90]).round(1).tolist(),
         'height_above_ground_of_bottom_p10_p50_p90': np.percentile([g['hag'] for g in R], [10, 50, 90]).round(2).tolist(),
         'bottom_z_sensor_frame_p5_p50_p95': np.percentile([g['bottom'] for g in R], [5, 50, 95]).round(2).tolist(),
         'outcomes': dict(Counter(g['outcome'] for g in R)),
         'wrong_class_as': dict(Counter(str(g['any_class']) for g in R if g['outcome'] == 'wrong_class'))}
    for name, key, bins in (('by_range', 'r', bins_r), ('by_points', 'n', bins_n)):
        tab = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            S = [g for g in R if lo <= g[key] < hi]
            if S:
                o = Counter(g['outcome'] for g in S)
                tab[f'{lo:g}-{hi:g}'] = {'n': len(S), **{k: o.get(k, 0) for k in ('hit', 'poor_box', 'wrong_class', 'missed')}}
        d[name] = tab
    pb = [g for g in R if g['outcome'] == 'poor_box']
    if pb:
        d['poor_box_errors'] = {'n': len(pb), 'median_iou3d': round(float(np.median([g['iou_same'] for g in pb])), 3),
                                'median_bev_iou': round(float(np.median([g['bev_same'] for g in pb])), 3),
                                'median_d_xy': round(float(np.median([g['d_xy'] for g in pb])), 2),
                                'median_d_z': round(float(np.median([g['d_z'] for g in pb])), 2),
                                'median_abs_d_z': round(float(np.median(np.abs([g['d_z'] for g in pb]))), 2),
                                'median_d_lwh': np.median([g['d_lwh'] for g in pb], 0).round(2).tolist(),
                                'median_d_yaw_deg': round(float(np.degrees(np.median([g['d_yaw'] for g in pb]))), 1),
                                'share_bev_ok_but_3d_fail': round(float(np.mean([g['bev_same'] >= 0.5 for g in pb])), 3)}
    yaw = np.array([g['box'][6] for g in R])
    d['heading_hist_30deg_bins_mod180'] = np.histogram(np.degrees(yaw) % 180, bins=6, range=(0, 180))[0].tolist()
    out['per_class'][c] = d

rel = {}
for name, kw in (('3d_iou0.5', dict(iou_thresholds=(0.5,), iou_type='3d')), ('3d_iou0.25', dict(iou_thresholds=(0.25,), iou_type='3d')),
                 ('bev_iou0.5', dict(iou_thresholds=(0.5,), iou_type='bev')), ('bev_iou0.25', dict(iou_thresholds=(0.25,), iou_type='bev'))):
    m = ev.evaluate_3d_detection(pred, gt, C, **kw)
    t = kw['iou_thresholds'][0]
    rel[name] = {'mAP': round(100 * m[f'mAP_{t}_R40'], 2), **{c: round(100 * m['per_class'][c][f'AP_{t}_R40'], 2) for c in C}}
out['relaxed_metrics'] = rel

out['false_positives_score>=0.3'] = {}
for c, L in fp.items():
    cells = Counter((round(x / 2) * 2, round(y / 2) * 2) for x, y, *_ in L)
    out['false_positives_score>=0.3'][c] = {'n': len(L), 'per_frame': round(len(L) / 50, 1),
                                           'top_2m_cells_xy_count': [[*k, v] for k, v in cells.most_common(8)],
                                           'frames_with_top_cell': len({l[4] for l in L if (round(l[0] / 2) * 2, round(l[1] / 2) * 2) == cells.most_common(1)[0][0]})}
# GT bottom plane (objects' own ground): fit over all GT bottoms
B = np.array([[g['box'][0], g['box'][1], g['bottom']] for g in gt_rows])
co = plane_fit(B)
out['gt_bottom_plane_z=ax+by+c'] = {'abc': co.round(4).tolist(), 'tilt_deg': round(float(np.degrees(np.arctan(np.hypot(co[0], co[1])))), 2),
                                    'residual_std_m': round(float((B[:, 2] - (B[:, 0] * co[0] + B[:, 1] * co[1] + co[2])).std()), 2),
                                    'bottom_z_range': [round(float(B[:, 2].min()), 2), round(float(B[:, 2].max()), 2)]}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
