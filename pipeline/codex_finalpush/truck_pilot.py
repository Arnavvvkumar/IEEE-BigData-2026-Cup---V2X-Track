"""Screen same-scan truck geometry and fixed confidence rules."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(BASE / 'codex_truck'), str(BASE / 'codex_postmortem')]
from rescorer import PASSED_MODELS, RESULTS, SOURCE, digest
from truck_replay import ap, calib, finalise, frame_rows, fusion, iou, read, screened
from audit import add_postprocess, copy_rule, prepared, score

MODES = ('median', 'visible_extent', 'point_support', 'agreement')


def truck_clusters(passes, ids, denominator):
    """Match the production seed-centred greedy truck clustering."""
    results = {}
    radius = fusion.RADIUS['truck'] * fusion.SCALE
    for frame in ids:
        items = sorted(((d, family) for family, pred in passes for d in pred[frame]
                        if d['class'] == 'truck'), key=lambda item: -item[0]['score'])
        if not items:
            results[frame] = []
            continue
        boxes = np.asarray([d['box'] for d, _ in items])
        scores = np.asarray([d['score'] for d, _ in items])
        unused = np.ones(len(items), dtype=bool)
        output = []
        for i, box in enumerate(boxes):
            if not unused[i]:
                continue
            distance = np.hypot(boxes[:, 0] - box[0], boxes[:, 1] - box[1])
            members = np.flatnonzero(unused & (distance < radius))
            unused[members] = False
            weights = scores[members]
            geometry = (boxes[members, :6] * weights[:, None]).sum(axis=0) / weights.sum()
            dimensions = boxes[members, 3:6]
            medians = []
            for column in dimensions.T:
                order = np.argsort(column, kind='stable')
                k = np.searchsorted(weights[order].cumsum(), weights.sum() / 2)
                medians.append(float(column[order[k]]))
            member_centers = boxes[members, :2]
            distances = np.linalg.norm(member_centers[:, None] - member_centers[None, :], axis=2)
            medoid = member_centers[int(np.argmin(distances @ weights))]
            output.append({'class': 'truck', 'score': float(min(weights.sum() / denominator, 1.)),
                           'box': geometry.tolist() + [float(box[6])], 'median': medians,
                           'families': len({items[j][1] for j in members}), 'medoid': medoid.tolist()})
        results[frame] = output
    return results


def update(records, cloud, mode):
    points = cloud[np.isfinite(cloud).all(axis=1) & (cloud != 0).any(axis=1)]
    tree = cKDTree(points[:, :2]) if mode in ('visible_extent', 'point_support') else None
    out = []
    for original in records:
        d = dict(original)
        if d['vehicle_copy']:
            out.append(d)
            continue
        if mode == 'median':
            d['box'] = d['box'][:3] + d['median'] + d['box'][6:]
        elif mode == 'center_medoid':
            d['box'] = d['medoid'] + d['box'][2:]
        elif mode == 'agreement':
            # A singleton is attenuated by 20%; heights of one checkpoint count once.
            d['score'] *= .8 if d['families'] == 1 else 1.
        elif tree is not None:
            box = np.asarray(d['box'])
            center, dims, yaw = box[:3], box[3:6], box[6]
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            idx = tree.query_ball_point(center[:2], np.linalg.norm(dims[:2]) / 2)
            local = (points[idx] - center) @ rotation.T
            inside = local[(abs(local) <= dims / 2).all(axis=1) & (local[:, 2] > -.5 * dims[2] + .12)]
            if mode == 'point_support':
                d['score'] *= .8 + .2 * min(len(inside) / 5, 1.)
            elif len(inside) >= 20:
                low, high = np.quantile(inside, [.05, .95], axis=0)
                extent = high - low
                # Preserve the prior on axes whose opposite face is poorly observed.
                visible = extent >= .6 * dims
                shift = .25 * (low + high) / 2 * visible
                dims = .75 * dims + .25 * np.where(visible, extent, dims)
                d['box'] = [*(center + shift @ rotation), *dims, yaw]
        out.append(d)
    return out


def postprocess(fused, clouds, modes):
    inputs = {f: [{**d, 'vehicle_copy': d['class'] == 'vehicle'} for d in records]
              for f, records in fused.items()}
    ruled = calib.truck_rule(inputs, **{**calib.TRUCK, 'a': .5, 'base': .1})
    for mode in modes:
        combinations = {}
        for frame, ds in ruled.items():
            trucks = [d for d in ds if d['class'] == 'truck']
            if mode != 'baseline':
                trucks = update(trucks, clouds[frame], mode)
            # Same dedupe and hedging as production, with shape changes preceding them.
            combinations[frame] = trucks
        yield mode, add_postprocess(calib.truck_dedupe(combinations))


def oracle(pred, labels):
    info = prepared(pred, labels)
    matrices = {frame: [{}, {}, {}] for frame in info}
    # Fix XY, dimensions, or both on the same highest-overlap primary match.
    for frame, (dets, truth, overlap, _) in info.items():
        for j, gt in enumerate(truth):
            priors = [i for i, d in enumerate(dets) if not d.get('hedge')]
            if not priors:
                continue
            i = max(priors, key=lambda k: overlap[k, j])
            if not .1 <= overlap[i, j] < .5:
                continue
            box, target = list(dets[i]['box']), gt['box']
            for k, span in enumerate(([0, 1], [3, 4, 5], [0, 1, 2, 3, 4, 5])):
                repaired = [target[h] if h in span else v for h, v in enumerate(box)]
                matrices[frame][k].setdefault(i, repaired)
    gains = []
    for k in range(3):
        changes = {}
        for frame, (dets, truth, old, _) in info.items():
            edits = matrices[frame][k]
            mat = old.copy()
            if edits:
                mat[list(edits)] = iou(list(edits.values()), [g['box'] for g in truth])
            changes[frame] = mat
        gains.append(score(info, changes)[0])
    return [score(info)[0], *gains]


def synthetic(modes, oracles=True):
    manifest = SOURCE / 'frames.json'
    records = {r['frame']: r for r in read(manifest) if r['split'] == 'validate'}
    ids = sorted(records)
    assert len(ids) == 500
    _, indices = next(GroupShuffleSplit(n_splits=1, test_size=.4, random_state=261114)
                      .split(ids, groups=[records[f]['layout_id'] for f in ids]))
    ids = sorted(ids[i] for i in indices)
    labels = {f: records[f]['ground_truth'] for f in ids}
    clouds = {}
    for f in ids:
        r = records[f]
        clouds[f] = np.fromfile(SOURCE / 'validate' / r['scenario'] / r['agent'] / (f + '.bin'),
                                dtype='<f4').reshape(-1, 4)[:, :3]
    passes = []
    for tag in PASSED_MODELS:
        path = RESULTS / f'{tag}_synthetic'
        assert digest(manifest) == read(path / 'result.json')['sha256']['source_manifest']
        preds = read(path / 'predictions.json')
        assert set(records) == set(preds)
        passes.append((tag, {f: preds[f] for f in ids}))
        raw = {f: [{**d, 'hedge': False} for d in preds[f] if d['class'] == 'truck'] for f in ids}
        if oracles:
            print(json.dumps({'oracle': tag, 'AP_baseline_XY_LWH_XYZLWH': oracle(raw, labels)}), flush=True)
    fusion.C, fusion.ONE_VOTE, fusion.YAW_CIRC = ['vehicle'], {'vehicle'}, {'vehicle'}
    vehicles = calib.calibrate(fusion.fuse([p for _, p in passes], ids), clouds)
    trucks = truck_clusters(passes, ids, len(passes))
    fused = {f: vehicles[f] + trucks[f] for f in ids}
    for mode, preds in postprocess(fused, clouds, ('baseline', *modes)):
        row = {'mode': mode, 'truck_AP': ap(frame_rows(preds, labels, 'truck'))}
        if mode == 'baseline' and oracles:
            row['oracle_AP_baseline_XY_LWH_XYZLWH'] = oracle(preds, labels)
        print(json.dumps(row), flush=True)


def development(size):
    """Check transfer on the archived v12 dev ensemble with exact postprocessing."""
    screened()
    dev = BASE / 'coop' / f'dev{size}'
    stem = 'v12p3x3_norule'
    gt = {f: [g for g in rows if g['points'] >= 5] for f, rows in read(dev / 'gt_coop.json').items()}
    cached = read(dev / f'pred_{stem}.json')
    clouds = fusion.load(dev / 'scans')
    ids = sorted(gt)
    assert set(ids) == set(cached) == set(clouds)
    spec = [line for line in (dev / 'eval_summary.txt').read_text().splitlines()
            if line.startswith(stem + ' | ')][-1].split(' | ')[1].removesuffix(' norule')
    passes, denominator = [], 0
    for model, classes in (token.split(':') for token in spec.split()):
        assert not model.startswith('second_twin_ll_')
        path = dev / 'allpred' / f'{model}_allpred.json'
        if 'truck' in classes.split(','):
            raw = read(path)
            denominator += len(raw)
            passes.extend((model, {f: [d for d in data[f] if d['class'] == 'truck']
                                   for f in ids}) for data in raw.values())
        else:
            # Cached dev head variants follow exactly the production 6/20 layout.
            denominator += 20 if '_tta_' in model else 6
    assert denominator == 178
    reconstructed = truck_clusters(passes, ids, denominator)
    for f, records in reconstructed.items():
        old = [d for d in cached[f] if d['class'] == 'truck']
        assert len(old) == len(records)
        for a, b in zip(old, records):
            np.testing.assert_array_equal(a['box'], b['box'])
            assert a['score'] == b['score']
    fixed, _ = finalise(cached)
    constant = {name: ap(frame_rows(fixed, gt, name)) for name in ('vehicle', 'pedestrian')}
    fused = {f: [d for d in cached[f] if d['class'] == 'vehicle'] + reconstructed[f] for f in ids}
    for mode, preds in postprocess(fused, clouds, ('baseline', *MODES)):
        official = ap(frame_rows(preds, gt, 'truck'))
        genuine = ap(frame_rows(preds, gt, 'truck', genuine=True))
        if mode == 'baseline':
            assert official == ap(frame_rows(fixed, gt, 'truck'))
        print(json.dumps({'set': size, 'mode': mode, **constant,
                          'truck': official, 'genuine_truck': genuine}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev', type=int, choices=(150, 350))
    parser.add_argument('--center-pilot', action='store_true')
    args = parser.parse_args()
    if args.center_pilot:
        synthetic(('center_medoid',), oracles=False)
    elif args.dev:
        development(args.dev)
    else:
        synthetic(MODES)
