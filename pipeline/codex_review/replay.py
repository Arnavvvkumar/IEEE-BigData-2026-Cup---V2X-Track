"""Replay final pedestrian fusion using screened development exports only."""
import csv
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path[:0] = [str(EXP / 'codex_truck'), str(EXP / 'final')]
from audit import KIT, iou, read, score, split
import calib
import detection_eval as ev
import fuse_test as fusion

P1 = 'pvrcnn_ped_P1_ep5'
P2 = 'pvrcnn_ped_P2_ep5'
V1 = 'pvrcnn_veh_V1_ep5'


def spec():
    base = re.search(r'^SPEC="([^"]+)"$', (EXP / 'final/build_candidate_v7.sh').read_text(), re.M)[1]
    result = [x for x in base.split() if x != 'second_twin_ll_s1_ep12:truck']
    result += ['pvrcnn_st_osc_ep6:pedestrian,vehicle,truck', 'pvrcnn_st_osc_s2_ep4:pedestrian,vehicle']
    return [tuple(x.split(':')) for x in result]


def screened():
    ref = ROOT / 'experiments/real_reference_audit'
    with (ref / 'v2x_real_forbidden_frames_mapping.csv').open(newline='') as f:
        ban = {r['forbidden_id']: r['point_cloud_file'].removesuffix('.bin') for r in csv.DictReader(f)}
    assert set(ban) == {f for f in (KIT / 'forbidden_frames.txt').read_text().splitlines()
                        if f and not f.startswith('#')} and len(ban) == 100
    seen = set()
    for path in [ref / 'real_dev_cw1/source_index.json'] + [
            EXP / 'coop' / f'dev{n}' / 'index.json' for n in (150,350)]:
        src = {v['source_key'] for v in read(path).values()}
        assert not src.intersection(seen) and not src.intersection(ban.values()), path
        seen |= src
    return ref / 'real_dev_cw1'


@njit
def matches(matrix):
    used = np.zeros(matrix.shape[1], dtype=np.bool_)
    outcomes = np.zeros(matrix.shape[0], dtype=np.bool_)
    for i in range(matrix.shape[0]):
        j_best, best = -1, 0.0
        for j in range(matrix.shape[1]):
            if not used[j] and matrix[i,j] > best:
                j_best, best = j, matrix[i,j]
        if best >= .5:
            outcomes[i], used[j_best] = True, True
    return outcomes


def metric(p, gt, c):
    prepared = []
    for f, detections in sorted(p.items()):
        ds = sorted(detections, key=lambda d: -d['score'])
        truth = [g for g in gt[f] if g['class'] == c]
        mat = iou([d['box'] for d in ds],[g['box'] for g in truth])
        prepared.append(([d['score'] for d in ds], matches(mat), len(truth)))
    rows = []
    for part in (prepared, prepared[::2], prepared[1::2]):
        score_arr = np.concatenate([r[0] for r in part])
        tp = np.concatenate([r[1] for r in part])[np.argsort(-score_arr)].cumsum()
        rows.append(round(100 * ev._compute_ap_40(
            tp / np.arange(1,len(tp)+1), tp / sum(r[2] for r in part)),5))
    return rows


def map_cache(clouds):
    dev = fusion.load(fusion.DEV)
    planes = np.array([fusion.fit_ground(p, np.random.default_rng(i)) for i,p in enumerate(dev.values())])
    R,t = fusion.level_transform(np.median(planes,0), fusion.GZ)
    def non_ground(p):
        q = p @ R.T + t
        return q[q[:,2] > fusion.GZ + fusion.GCUT]
    keys, count = np.unique(np.concatenate([np.unique(fusion.vkey(non_ground(p))) for p in dev.values()]),
                            return_counts=True)
    static = keys[count >= fusion.FRAC * len(dev)]
    def build_cache(p):
        level = non_ground(p)
        calibration = calib.nonground(p)
        return (level, np.isin(fusion.vkey(level),static), calibration,
                cKDTree(level[:,:2]), cKDTree(calibration[:,:2]))
    return R, t, {f:build_cache(p) for f,p in clouds.items()}


def calibrate_hedge(fused, cache, down, center_factor, floor):
    R,t,frames = cache
    out = {}
    for f, dets in fused.items():
        p, mask, calibration_points, map_tree, calibration_tree = frames[f]
        frame = []
        boxes = np.array([d['box'] for d in dets]).reshape(-1,7)
        centers = boxes[:,:3] @ R.T + t
        radius = np.linalg.norm(boxes[:,3:5],axis=1)/2
        map_hits = map_tree.query_ball_point(centers[:,:2],radius,return_sorted=True)
        for d,b,center,local in zip(dets,boxes,centers,map_hits):
            h = R @ [np.cos(b[6]),np.sin(b[6]),0.]
            m = fusion.in_box(p[local],center,b[3:6],np.arctan2(h[1],h[0]))
            sf = mask[local][m].mean() if m.any() else 0.
            out_score = d['score'] * (1 - down * sf)
            nearby = calibration_tree.query_ball_point(b[:2],1.1,return_sorted=True)
            b = calib.refine_ped(b, calibration_points[nearby],
                **{**calib.PED, 's':calib.PED['s']*center_factor, 'c':calib.PED['c']*center_factor})
            frame.append({'class':'pedestrian','score':out_score,'box':b.tolist()})
        out[f] = frame
    return calib.hedge_pedestrians(out,smin=floor)


def main(size):
    screened()
    start = time.perf_counter()
    d = EXP / 'coop' / f'dev{size}'
    gt = {f:[g for g in truth if g['points'] >= 5] for f,truth in read(d/'gt_coop.json').items()}
    ids = sorted(gt)
    base = spec()
    counts, active = {}, {}
    for tag, classes in base + [(P1,'pedestrian'),(P2,'pedestrian'),(V1,'vehicle')]:
        n = d / 'allpred' / (tag + '_allpred.json')
        raw = read(n)
        counts[tag] = len(raw)
        if 'pedestrian' in classes:
            assert all(set(a)==set(ids) for a in raw.values()), n
            active[tag] = [{f:[x for x in a[f] if x['class']=='pedestrian'] for f in ids}
                           for a in raw.values()]
        del raw
        gc.collect()
    clouds = fusion.load(d/'scans')
    cache = map_cache(clouds)
    del clouds
    fusion.C = ['pedestrian']
    fusion.ONE_VOTE = {'vehicle'}
    fusion.YAW_CIRC = {'vehicle'}
    # Prespecify a small grid. P2 x3 (v11 proposal) is the reference.
    # Tuple: label, P1 weight, P2 weight, pedestrian radius (m), static scale,
    # center adjustment multiplier for s and c, hedge eligibility floor.
    options = [
        ('p2x3',3,3,.24,.8,1.,.05), ('p2x0',3,0,.24,.8,1.,.05),
        ('p2x1',3,1,.24,.8,1.,.05), ('p2x2',3,2,.24,.8,1.,.05),
        ('p1x2',2,3,.24,.8,1.,.05),
        ('radius20',3,3,.20,.8,1.,.05), ('radius28',3,3,.28,.8,1.,.05),
        ('static65',3,3,.24,.65,1.,.05), ('static95',3,3,.24,.95,1.,.05),
        ('center80',3,3,.24,.8,.8,.05), ('center120',3,3,.24,.8,1.2,.05),
        ('floor04',3,3,.24,.8,1.,.04), ('floor06',3,3,.24,.8,1.,.06),
    ]
    fused_cache = {}
    for name,w1,w2,r,down,center,floor in options:
        key = w1,w2,r
        if key not in fused_cache:
            n = sum(counts[tag] for tag,_ in base) + w1*counts[P1] + w2*counts[P2]
            p = [x for tag,cls in base if 'pedestrian' in cls for x in active[tag]] + active[P1]*w1 + active[P2]*w2
            fusion.RADIUS['pedestrian'] = r/fusion.SCALE
            fused_cache[key] = fusion.fuse(p + [dict.fromkeys(ids, ())]*(n-len(p)),ids)
        out = calibrate_hedge(fused_cache[key],cache,down,center,floor)
        row = metric(out,gt,'pedestrian')
        if name == 'p2x3':
            sub = ids[::40]
            exact = ev.evaluate_3d_detection(split(out,sub),split(gt,sub),['pedestrian'],(.5,))['mAP_0.5_R40']
            assert np.isclose(metric(split(out,sub),gt,'pedestrian')[0], exact*100)
        print(json.dumps({'set':size,'candidate':name,'AP':row,'seconds':round(time.perf_counter()-start,1)}),flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
