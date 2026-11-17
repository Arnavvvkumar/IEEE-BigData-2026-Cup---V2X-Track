"""Read-only truck diagnostics. Test archives are compared for integrity only."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
KIT = ROOT / 'starting_kit/track_b'
sys.path[:0] = [str(EXP / 'final'), str(KIT / 'scoring_program')]
import calib
import detection_eval as ev

DATA = Path.home() / 'projects/bdc2026-track-b/data'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(boxes, truths):
    """The kit's oriented BEV intersection times vertical overlap."""
    a = np.asarray(boxes, dtype=float).reshape(-1, 7)
    b = np.asarray(truths, dtype=float).reshape(-1, 7)
    m = np.zeros((len(a), len(b)))
    if not m.size:
        return m
    d = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2)
    radius = lambda u: np.linalg.norm(u[:, 3:5], axis=1) / 2
    az, ah = a[:, 2], a[:, 5] / 2
    bz, bh = b[:, 2], b[:, 5] / 2
    z = np.minimum((az + ah)[:, None], bz + bh) - np.maximum((az - ah)[:, None], bz - bh)
    x, y = np.where((d <= radius(a)[:, None] + radius(b)) & (z > 0))
    if len(x):
        polygon = lambda p: shapely.polygons(np.array([ev._box_corners_2d(q) for q in p]))
        overlap = shapely.area(shapely.intersection(polygon(a)[x], polygon(b)[y])) * z[x, y]
        m[x, y] = overlap / (a[x, 3:6].prod(1) + b[y, 3:6].prod(1) - overlap)
    return m


def split(records, ids):
    return {f: records[f] for f in ids}


def copy_rule(pred, a=0.5):
    # calib.truck_rule preserves extra keys. Retain source flags for diagnosis.
    tagged = {f: [{**d, 'vehicle_copy': d['class'] == 'vehicle'}
                  for d in v if d['class'] in ('truck', 'vehicle')] for f, v in pred.items()}
    ruled = calib.truck_dedupe(calib.truck_rule(tagged, **{**calib.TRUCK, 'a': a, 'base': 0.1}))
    return {f: [d for d in v if d['class'] == 'truck'] for f, v in ruled.items()}


def add_postprocess(pred, directory=None, boost=1.0):
    # Same geom and score as calib.truck_boost_hedge. For the unboosted build,
    # no point reads are necessary. For a boost reuse the exact counting path.
    out = {}
    for f, dets in pred.items():
        if directory is not None:
            cloud = np.fromfile(directory / f'{f}.bin', dtype='<f4').reshape(-1, 3)
            pp = calib.truck_boost_hedge({f: dets}, {f: cloud}, boost=boost)[f]
        else:
            pp = []
            for d in dets:
                b = d['box']
                pp.append(d)
                if d['score'] >= .02:
                    for length, width, height in calib.TRUCK_HEDGE:
                        dz = b[5] * height
                        pp.append({**d, 'box': [b[0], b[1], b[2] + (dz-b[5])/2,
                                                b[3]*length, b[4]*width, dz, b[6]],
                                   'score': d['score'] * .05})
        primary = [i for d in dets for i in [True] + ([False]*len(calib.TRUCK_HEDGE) if d['score'] >= .02 else [])]
        out[f] = [{**d, 'hedge': not p} for d, p in zip(pp, primary, strict=True)]
    return out


def prepared(pred, gt, genuine=False):
    info = {}
    for f, labels in gt.items():
        trucks = [g for g in labels if g['class'] == 'truck' and (not genuine or g.get('type') != 'TrashCan')]
        cars = [g['box'] for g in labels if g['class'] == 'vehicle']
        can = [g['box'] for g in labels if g.get('type') == 'TrashCan'] if genuine else []
        dets = sorted(pred[f], key=lambda d: -d['score'])
        ignored = iou([d['box'] for d in dets], can)
        if ignored.size:
            dets = [d for d, ignore in zip(dets, (ignored >= .3).any(1)) if not ignore]
        boxes = [d['box'] for d in dets]
        info[f] = (dets, trucks, iou(boxes, [g['box'] for g in trucks]), cars)
    return info


def score(info, replacements=None):
    scores, matched, n_gt = [], [], 0
    for f, (dets, truck, mat, _) in info.items():
        if replacements and f in replacements:
            mat = replacements[f]
        n_gt += len(truck)
        used = np.zeros(len(truck), dtype=bool)
        for row, d in zip(mat, dets):
            if len(row):
                i = np.where(used, -1, row).argmax()
                match = row[i] >= .5 and not used[i]
                if match:
                    used[i] = True
            else:
                match = False
            scores.append(d['score'])
            matched.append(match)
    seq = np.asarray(matched)[np.argsort(-np.asarray(scores))]
    tp = seq.cumsum()
    return 100 * ev._compute_ap_40(tp / np.arange(1, len(tp)+1), tp / n_gt), int(tp[-1])


def diagnose(info, raw):
    totals, long, fp, oracle, fp_by_frame = Counter(), Counter(), Counter(), {}, Counter()
    geometry_edits = ('xy', 'yaw', 'vertical', 'extents', 'all')
    mats = {k: {} for k in geometry_edits}
    n_gt = sum(len(t) for _, t, _, _ in info.values())
    n_attainable = 0
    for f, (dets, truth, m, cars) in info.items():
        if m.size:
            n_attainable += int((maximum_bipartite_matching(csr_matrix(m >= .5), perm_type='column') >= 0).sum())
        b = [d['box'] for d in dets]
        by_class = {c: [d['box'] for d in raw[f] if d['class'] == c] for c in ('truck', 'vehicle')}
        truth_boxes = [g['box'] for g in truth]
        rv = iou(by_class['vehicle'], truth_boxes).max(0, initial=0)
        rt = iou(by_class['truck'], truth_boxes).max(0, initial=0)
        for j, g in enumerate(truth):
            gt_box = g['box']
            target = long if gt_box[3] >= 7.5 else None
            col = m[:, j]
            mx = col.max(initial=0)
            label = 'matchable' if mx >= .5 else 'loose' if mx >= .1 else 'no_overlap'
            for counts in (totals, target):
                if counts is None:
                    continue
                counts[label] += 1
                if (rt[j] < .5) and rv[j] >= .5:
                    counts['vehicle_only_raw_iou50'] += 1
                    counts['vehicle_only_raw_and_recovered'] += int(mx >= .5)
            if label == 'loose':
                # One best existing box per loose GT. Keep the exact scores,
                # exclude candidates able to match another GT already.
                eligible = np.where(m.max(axis=1) < .5, col, -1)
                i = eligible.argmax()
                if eligible[i] < .1:
                    continue
                current = b[i]
                for k, indices in zip(geometry_edits, ([0,1], [6], [2,5], [3,4,5], range(7))):
                    new = current.copy()
                    for ix in indices:
                        new[ix] = gt_box[ix]
                    ma = mats[k].setdefault(f, m.copy())
                    # Keep the better result only in this privileged diagnostic.
                    u = iou([new], truth_boxes)[0]
                    if u[j] >= .5 and u[j] > ma[i, j]:
                        ma[i] = u
        if dets:
            hit = m.max(1, initial=0) >= .5
            # Count guaranteed FPs (IoU < .5 to every truck) above .15,
            # diagnostic only. Duplicates are measured separately in AP.
            bad = [i for i, d in enumerate(dets) if d['score'] >= .15 and not hit[i]]
            wrong = iou([b[i] for i in bad], cars).max(1, initial=0) >= .5
            for i, is_car in zip(bad, wrong):
                fp['iou_below50'] += 1
                fp['vehicle_copy'] += int(dets[i]['vehicle_copy'])
                fp['matches_vehicle_gt'] += int(is_car)
            fp_by_frame[len(bad)] += 1
    for k, corrected in mats.items():
        oracle[k] = round(score(info, corrected)[0], 4)
    ap, tp = score(info)
    return {'n_gt': n_gt, 'AP': round(ap, 4), 'greedy_tp': tp, 'ranking_ceiling_AP': 2.5*(40*n_attainable//n_gt),
            'max_simultaneous_matches': n_attainable, 'outcomes': dict(totals), 'long_outcomes': dict(long),
            'high_fp': dict(fp), 'high_fp_per_frame_hist': dict(fp_by_frame), 'geometry_oracles_AP': oracle}


def dev():
    audit = ROOT / 'experiments/real_reference_audit'
    with (audit / 'v2x_real_forbidden_frames_mapping.csv').open(newline='') as inp:
        mapping = {r['forbidden_id']: r['point_cloud_file'].removesuffix('.bin') for r in csv.DictReader(inp)}
    forbidden = {x for x in (KIT / 'forbidden_frames.txt').read_text().splitlines() if x and not x.startswith('#')}
    assert len(forbidden) == 100 and set(mapping) == forbidden
    static_index = read(audit / 'real_dev_cw1/source_index.json')
    assert len(static_index) == 50 and not {r['source_key'] for r in static_index.values()}.intersection(mapping.values())
    seen = set()
    for size in (150, 350):
        d = EXP / 'coop' / f'dev{size}'
        index = read(d / 'index.json')
        src = {r['source_key'] for r in index.values()}
        assert len(src) == size and not src.intersection(mapping.values()) and not src.intersection(seen)
        seen.update(src)
        gt = {f: [g for g in truth if g['points'] >= 5] for f, truth in read(d / 'gt_coop.json').items()}
        pred = read(d / 'pred_v10_norule.json')
        assert set(gt) == set(pred) == set(index)
        base = copy_rule(pred)
        variants = {'base': add_postprocess(base),
                    'v10t1': add_postprocess(base, d / 'scans', boost=2.0),
                    'v10t2': add_postprocess(copy_rule(pred, .3))}
        one, first = {'zero_forbidden': len(index)}, True
        for label, v in variants.items():
            row = {}
            for genuine in (False, True):
                info = prepared(v, gt, genuine)
                ids = sorted(info)
                k = 'genuine' if genuine else 'all'
                row[k] = [round(score(split(info, ii))[0], 4) for ii in (ids, ids[::2], ids[1::2])]
                if first:
                    one[k + '_diagnostic'] = diagnose(info, pred)
                    if not genuine:
                        # Verify full evaluator equivalence on an independent small subset.
                        sample = ids[::25]
                        check = ev.evaluate_3d_detection(split(v, sample), split(gt, sample), ['truck'], (0.5,))
                        assert np.isclose(100 * check['mAP_0.5_R40'], score(split(info, sample))[0])
            one[label] = row
            first = False
        print(json.dumps({f'dev{size}': one}), flush=True)


def synthetic():
    source = DATA / 'claude_v2_scale5500_fix1'
    res = ROOT / 'experiments/detector_program/overnight_accuracy_results/overnight_control_seed1234_synthetic'
    assert digest(source / 'frames.json') == read(res / 'result.json')['sha256']['source_manifest']
    gt = {r['frame']: r['ground_truth'] for r in read(source / 'frames.json') if r['split'] == 'validate'}
    raw = read(res / 'predictions.json')
    assert len(gt) == 500 and set(gt) == set(raw)
    p = {f: [{**d, 'vehicle_copy': False} for d in v if d['class'] == 'truck'] for f, v in raw.items()}
    base, corrected = prepared(p, gt), prepared(add_postprocess(copy_rule(raw)), gt)
    assert np.isclose(score(base)[0], 100 * read(res / 'result.json')['metrics']['per_class']['truck']['AP_0.5_R40'])
    print(json.dumps({'single_SECOND_native': diagnose(base, raw),
                      'single_SECOND_with_rule_and_hedge': diagnose(corrected, raw)}), flush=True)


def sizes():
    for name in ('claude_canon_all_v1', 'claude_twin_v1', 'claude_twin_st_v1', 'claude_twin_pv_v1', 'claude_twin_ll_v1'):
        p = DATA / name
        ids = (p / 'ImageSets/train.txt').read_text().split()
        boxes = np.array([[float(x) for x in line.split()[3:6]] for f in ids
                          for line in (p / 'labels' / (f + '.txt')).read_text().splitlines()
                          if line.split() and line.split()[7] == 'truck'])
        print(name, len(ids), len(boxes), 'median LWH', np.median(boxes, 0).round(2).tolist(),
              'long-low %', round(100 * ((boxes[:, 0] >= 7.5) & (boxes[:, 2] <= 2.9)).mean(), 2), flush=True)


def archives():
    def load(tag):
        file = ROOT.parent / f'bdc2026_trackB_submission_{tag}.zip'
        with ZipFile(file) as z:
            pred = json.loads(z.read('predictions.json'))
            assert len(pred) == 50
            syn = {i.filename: hashlib.sha256(z.read(i)).hexdigest()
                   for i in z.infolist() if i.filename.startswith('synthetic/') and i.filename.endswith('.bin')}
            return pred, syn, sum(i.file_size for i in z.infolist())
    base, bs, _ = load('v10r')
    for tag in ('v10t1','v10t2'):
        p, syn, size = load(tag)
        assert set(p) == set(base)
        def items(obj, c, scored=True):
            return [(f, tuple(x['box']), x['score'] if scored else None)
                    for f, ds in sorted(obj.items()) for x in ds if x['class'] == c]
        for c in ('vehicle', 'pedestrian', 'truck'):
            a, b = items(base, c), items(p, c)
            ca, cb = Counter(a), Counter(b)
            print(tag,c,'unchanged_in_order',a==b,'scored_removed',sum((ca-cb).values()),
                  'scored_added',sum((cb-ca).values()), 'same_geometry',
                  Counter(items(base,c,False))==Counter(items(p,c,False)))
        print(tag,'same_realism_files',syn==bs,'uncompressed_bytes',size)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
