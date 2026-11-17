"""Synthetic-only feasibility experiment using cached detector exports.

Each cached model supplies one pass. This reproduces the fusion and truck
rule, but does not reproduce the production six-height ensemble. The
held-out group is fixed by layout before fitting. No deployment is implied.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

from audit import DATA, EXP, ROOT, add_postprocess, copy_rule, digest, iou, prepared, read, score

import fuse_test as fusion

SOURCE = DATA / 'claude_v2_scale5500_fix1'
RESULTS = ROOT / 'experiments/detector_program/overnight_accuracy_results'
PASSED_MODELS = ('overnight_control_seed1234', 'relabel_containers_seed1234', 'pvrcnn_relabel_seed1234')
HERE = Path(__file__).resolve().parent


def features(dets, cloud):
    cloud = cloud[np.isfinite(cloud).all(1) & (np.abs(cloud).sum(1) > 0)]
    pts = []
    for d in dets:
        x, y, z, length, width, height, yaw = d['box']
        local = cloud[np.hypot(cloud[:, 0]-x, cloud[:, 1]-y) <= np.hypot(length, width)/2] - [x,y,z]
        picked = local[fusion.in_box(local, np.zeros(3), [length,width,height], yaw)]
        n = len(picked)
        hist = np.histogram(picked[:, 2] / height, bins=np.linspace(-.5,.5,5))[0] / max(n,1)
        pts.append([d['score'], length, width, height, np.hypot(x,y), np.log1p(n),
                    np.log1p(n / (length*width*height)), *hist,
                    int(d['vehicle_copy']), int(d['hedge'])])
    return np.asarray(pts, dtype=float).reshape(-1,13)


def main():
    start = time.perf_counter()
    manifest = SOURCE / 'frames.json'
    records = {r['frame']: r for r in read(manifest) if r['split'] == 'validate'}
    ids = sorted(records)
    assert len(ids) == 500
    groups = [records[f]['layout_id'] for f in ids]
    train_idx, eval_idx = next(GroupShuffleSplit(n_splits=1, test_size=.4, random_state=261114)
                               .split(ids, groups=groups))
    train, heldout = ([ids[j] for j in part] for part in (train_idx, eval_idx))
    assert not {records[f]['layout_id'] for f in train}.intersection(records[f]['layout_id'] for f in heldout)

    passes = []
    for tag in PASSED_MODELS:
        path = RESULTS / f'{tag}_synthetic'
        assert digest(manifest) == read(path / 'result.json')['sha256']['source_manifest']
        pred = read(path / 'predictions.json')
        assert set(pred) == set(ids)
        passes.append({f:[d for d in boxes if d['class'] in ('truck','vehicle')]
                       for f,boxes in pred.items()})
    fusion.YAW_CIRC, fusion.ONE_VOTE, fusion.TRUCK_SCORE = {'vehicle'}, {'vehicle'}, ''
    fused = fusion.fuse(passes, ids)
    # Vehicle calibration used by the production build. Here it is part of
    # the fixed baseline. The labels and learned parameters come only from
    # the synthetic fit split.
    for ds in fused.values():
        for d in ds:
            if d['class'] == 'vehicle':
                d['box'][2] += .11
                d['box'][5] += .09
    base = add_postprocess(copy_rule(fused))
    gt = {f: records[f]['ground_truth'] for f in ids}
    feat = {}
    for f in ids:
        r = records[f]
        path = SOURCE / 'validate' / r['scenario'] / r['agent'] / (f + '.bin')
        cloud = np.fromfile(path, dtype='<f4').reshape(-1,4)[:, :3]
        feat[f] = features(base[f], cloud)

    # Train positive if a prediction could match a synthetic truck at .5.
    # Duplicates still count as false positives under the official scorer.
    def target(f):
        truck = [g['box'] for g in gt[f] if g['class'] == 'truck']
        return iou([d['box'] for d in base[f]], truck).max(1, initial=0) >= .5

    X = np.concatenate([feat[f] for f in train])
    y = np.concatenate([target(f) for f in train])
    model = HistGradientBoostingClassifier(max_iter=90, max_leaf_nodes=15, l2_regularization=2,
                                           random_state=261114, early_stopping=False)
    model.fit(X, y)

    def metric(pred):
        return score(prepared(pred, {f:gt[f] for f in heldout}))[0]

    # Per-size fit from synthetic training data. Only long, primary
    # predictions with IoU >= .30 inform the median extent and z residual.
    residuals = []
    for f in train:
        ds = [d for d in base[f] if not d['hedge'] and d['box'][3] >= 7.5 and d['score'] >= .02]
        truck = [g['box'] for g in gt[f] if g['class'] == 'truck']
        m = iou([d['box'] for d in ds], truck)
        for d,row in zip(ds,m):
            if row.max(initial=0) >= .30:
                b,g = np.asarray(d['box']), np.asarray(truck[row.argmax()])
                residuals.append([g[2]-b[2], *(g[3:6] / b[3:6])])
    dz, l, w, h = np.median(residuals, axis=0)
    geo = {}
    rank = {}
    for f in heldout:
        pred = base[f]
        q = model.predict_proba(feat[f])[:,1] if pred else []
        # Deterministic score ordering when tree leaves tie.
        rank[f] = [{**d, 'score': float(s + 1e-9*d['score'])} for d,s in zip(pred,q)]
        geo[f] = []
        for d in pred:
            b = list(d['box'])
            if b[3] >= 7.5 and not d['hedge']:
                b[2] += dz
                b[3] *= l
                b[4] *= w
                b[5] *= h
            geo[f].append({**d, 'box':b})

    result = {'models_one_pass_each': PASSED_MODELS, 'fit_frames':len(train), 'heldout_frames':len(heldout),
              'fit_examples': len(y), 'fit_positives':int(y.sum()), 'fit_long_boxes':len(residuals),
              'long_geo_dz_and_lwh_scales':[dz,l,w,h], 'baseline_AP':metric(base),
              'rescoring_AP':metric(rank), 'size_geometry_AP':metric(geo),
              'heldout_gt': sum(g['class']=='truck' for f in heldout for g in gt[f]),
              'wall_seconds': time.perf_counter()-start,
              'fit_ids_sha256':hashlib.sha256('\n'.join(train).encode()).hexdigest(),
              'heldout_ids_sha256':hashlib.sha256('\n'.join(heldout).encode()).hexdigest()}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
