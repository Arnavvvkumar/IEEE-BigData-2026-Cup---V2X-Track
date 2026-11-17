"""Label-free final fusion for a folder of scans (the 50 released test scans): the settings fixed on the dev frames.
 1. box fusion over (model, ground height) passes: greedy BEV-centre clustering, score-weighted mean box, score = sum/passes
    (canon/fuse.py 'mean', radius scale 0.6); pass spec <allpred.json>[:g1,g2[:class1,class2]].
 2. pedestrian score *= (1 - 0.8 * static fraction), static map = 0.25 m voxels occupied in >= 30% of the UNLABELLED dev frames
    (levelled, non-ground points); twin/static_filter.py variant ped_scale0.8_f0.3. The map never contains the scored scans.
usage: fuse_test.py <scans dir> <out predictions.json> <pass spec> ..."""
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
sys.path.insert(0, str(W / 'experiments/claude_carla_overnight/erroranalysis'))
from groundplane import fit_ground, level_transform  # noqa: E402

DEV = W / 'experiments/real_reference_audit/real_dev_cw1/detection_test_frames'
C = ['vehicle', 'pedestrian', 'truck']
RADIUS = {'vehicle': 1.0, 'pedestrian': 0.4, 'truck': 1.5}
SCALE, V, GZ, GCUT, FRAC, DOWN = 0.6, 0.25, -3.0, 0.25, 0.3, 0.8
# Classes whose fused heading is the weighted circular mean of the cluster instead of the top box's heading.
# Default empty = the v3 behaviour. Set e.g. FUSE_YAW_CIRC=vehicle (dev150/dev350 vehicle AP 74.1->78.6 / 73.9->78.0).
import os  # noqa: E402
YAW_CIRC = set(filter(None, os.environ.get('FUSE_YAW_CIRC', '').split(',')))
# Experiment only (default empty = unchanged): FUSE_TRUCK_SCORE=max:k | mean:k | geo:k  -> fused truck score = k * (max | mean | sqrt(sum*max))
TRUCK_SCORE = os.environ.get('FUSE_TRUCK_SCORE', '')
# FUSE_ONE_VOTE=vehicle,truck: one confidence vote per pass per cluster (Codex found +0.5/+0.3 vehicle AP on dev150/dev350)
ONE_VOTE = set(filter(None, os.environ.get('FUSE_ONE_VOTE', '').split(',')))
# Experiment overrides (unset = unchanged defaults): pedestrian cluster radius (m, before SCALE) and static down-weight strength.
if os.environ.get('FUSE_PED_RADIUS'):
    RADIUS['pedestrian'] = float(os.environ['FUSE_PED_RADIUS'])
if os.environ.get('FUSE_STATIC_DOWN'):
    DOWN = float(os.environ['FUSE_STATIC_DOWN'])
# Extra classes added to YAW_CIRC (build scripts set FUSE_YAW_CIRC=vehicle inline; v16 adds pedestrian: P35 dev150 +0.7, dev350 +0.2).
YAW_CIRC |= set(filter(None, os.environ.get('FUSE_YAW_CIRC_EXTRA', '').split(',')))
# Experiment only (unset = unchanged): pedestrian box GEOMETRY (centre/size/heading) from weights score**FUSE_PED_WPOW, and/or only the
# FUSE_PED_TOPK highest-scoring members of the cluster. The fused score is unchanged.
PED_WPOW = float(os.environ.get('FUSE_PED_WPOW', '1'))
PED_TOPK = int(os.environ.get('FUSE_PED_TOPK', '0'))
# Experiment only (unset = unchanged; Codex's idea): before averaging, rewrite each member of a cluster of these classes in the
# equivalent form (dx, dy, yaw) == (dy, dx, yaw +- 90 deg) whose heading is closest to the top member's, so the mean does not mix the
# two descriptions of one footprint. Every individual box occupies exactly the same space after the rewrite.
ALIGN = set(filter(None, os.environ.get('FUSE_ALIGN', '').split(',')))
AUDIT = {'members': 0, 'rewritten': 0, 'clusters': 0, 'clusters_rewritten': 0, 'ped_clusters': 0, 'ped_bimodal': 0}
if os.environ.get('FUSE_AUDIT'):
    import atexit
    atexit.register(lambda: Path(os.environ['FUSE_AUDIT']).write_text(json.dumps(AUDIT)))


def align(gb):
    gb = gb.copy()
    d = (gb[:, 6] - gb[0, 6] + np.pi / 2) % np.pi - np.pi / 2      # heading difference to the top member, mod 180, in [-90, 90)
    sw = np.abs(d) > np.pi / 4
    gb[sw, 3], gb[sw, 4] = gb[sw, 4].copy(), gb[sw, 3].copy()
    gb[sw, 6] -= np.sign(d[sw]) * np.pi / 2
    return gb, int(sw.sum())


# Codex ideas 1/2 (experiment only; unset = unchanged). PASS_GROUP[k] = checkpoint name of pass k (TTA grouped with its original),
# set by the caller. FUSE_PED_AGREE=lambda: score *= 1 - lambda*(1 - median 3D IoU of per-checkpoint representative boxes vs the
# fused box); FUSE_PED_ZMED=a: fused z/height = a * (weighted-median bottom/top of the representatives) + (1-a) * current.
# Both abstain with fewer than 3 checkpoints. Representative = the checkpoint's highest-scored member in the cluster.
PASS_GROUP = None
PED_AGREE = float(os.environ.get('FUSE_PED_AGREE', '0'))
PED_ZMED = float(os.environ.get('FUSE_PED_ZMED', '0'))
# Codex idea (Nov 15 PM, experiment; unset = unchanged): FUSE_PED_MODE=1|2, FUSE_PED_MODE_D=sep (m). If a pedestrian cluster's per-checkpoint
# representatives split into two BEV groups (2-means) whose centres are > sep apart, each backed by >= 2 checkpoints, then
# mode 1: box geometry (xyz, lwh) = score-weighted mean of the members of the group backed by more checkpoints (tie: higher score sum);
# mode 2: xy = 50/50 blend of that group's mean and the original fused xy. Score and yaw unchanged.
PED_MODE = int(os.environ.get('FUSE_PED_MODE', '0'))
PED_MODE_D = float(os.environ.get('FUSE_PED_MODE_D', '0.35'))
_EV = None


def iou3d(a, b):
    global _EV
    if _EV is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location('ev', W / 'starting_kit/track_b/scoring_program/detection_eval.py')
        _EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(_EV)
    return float(_EV._iou_3d(np.asarray(a, float), np.asarray(b, float)))


def rep_boxes(mi, pid, boxes, scores):
    """one row [box(7), score] per checkpoint: its highest-scored member (mi is in descending score order)"""
    seen = {}
    for j in mi:
        g = PASS_GROUP[pid[j]]
        if g not in seen:
            seen[g] = np.r_[boxes[j, :7], scores[j]]
    return np.array(list(seen.values()))


def bimodal_pick(mi, pid, boxes, scores):
    """indices (into boxes) of the members of the better-supported BEV group, or None if the cluster is not clearly bimodal"""
    reps = {}
    for j in mi:   # descending score: first member seen per checkpoint is its representative
        g = PASS_GROUP[pid[j]]
        if g not in reps:
            reps[g] = (boxes[j, :2], scores[j])
    if len(reps) < 4:
        return None
    keys = list(reps)
    xy = np.array([reps[k][0] for k in keys]); sw = np.array([reps[k][1] for k in keys])
    dm = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    a, b = np.unravel_index(np.argmax(dm), dm.shape)
    ca, cb = xy[a].copy(), xy[b].copy()
    for _ in range(10):
        lab = np.hypot(*(xy - ca).T) > np.hypot(*(xy - cb).T)   # True = group b
        if lab.all() or (~lab).all():
            return None
        ca = (xy[~lab] * sw[~lab, None]).sum(0) / sw[~lab].sum()
        cb = (xy[lab] * sw[lab, None]).sum(0) / sw[lab].sum()
    if np.hypot(*(ca - cb)) <= PED_MODE_D or min(lab.sum(), (~lab).sum()) < 2:
        return None
    na, nb = (~lab).sum(), lab.sum()
    pick_b = nb > na or (nb == na and sw[lab].sum() > sw[~lab].sum())
    chosen = {k for k, l in zip(keys, lab) if l == pick_b}
    return np.array([j for j in mi if PASS_GROUP[pid[j]] in chosen])


def wmedian(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, c[-1] / 2)])


def load(d):
    return {f.stem: np.fromfile(f, dtype='<f4').reshape(-1, 3) for f in sorted(Path(d).glob('*.bin'))}


def fuse(passes, ids):
    n, out = len(passes), {}
    for f in ids:
        dets = []
        for c in C:
            items = sorted(((d, k) for k, p in enumerate(passes) for d in p[f] if d['class'] == c), key=lambda t: -t[0]['score'])
            if not items:
                continue
            pid = np.array([k for _, k in items])
            items = [d for d, _ in items]
            boxes = np.array([d['box'] for d in items])
            scores = np.array([d['score'] for d in items])
            used = np.zeros(len(items), bool)
            r = RADIUS[c] * SCALE
            for i in range(len(items)):
                if used[i]:
                    continue
                member = (~used) & (np.hypot(boxes[:, 0] - boxes[i, 0], boxes[:, 1] - boxes[i, 1]) < r)
                used |= member
                w = scores[member]
                gb, gw = boxes[member], w
                if c == 'pedestrian' and (PED_WPOW != 1 or PED_TOPK):
                    if PED_TOPK:   # members are in descending score order
                        gb, gw = gb[:PED_TOPK], gw[:PED_TOPK]
                    gw = gw ** PED_WPOW
                if c in ALIGN or os.environ.get('FUSE_AUDIT'):
                    ab, k = align(gb)
                    if c == 'pedestrian':
                        AUDIT['members'] += len(gb); AUDIT['rewritten'] += k; AUDIT['clusters'] += 1; AUDIT['clusters_rewritten'] += k > 0
                    if c in ALIGN:
                        gb = ab
                b = (gb[:, :6] * gw[:, None]).sum(0) / gw.sum()
                if c in YAW_CIRC:  # score-weighted circular mean heading, period pi (a box and its 180-degree flip are the same box)
                    a = 2 * gb[:, 6]
                    yaw = float(np.arctan2((gw * np.sin(a)).sum(), (gw * np.cos(a)).sum()) / 2)
                else:
                    yaw = float(boxes[i, 6])
                if c == 'pedestrian' and PED_ZMED and PASS_GROUP is not None:   # Codex idea 2: robust vertical fusion
                    rb = rep_boxes(np.flatnonzero(member), pid, boxes, scores)
                    if len(rb) >= 3:
                        bt, tp, ws = rb[:, 2] - rb[:, 5] / 2, rb[:, 2] + rb[:, 5] / 2, rb[:, 7]
                        bm, tm = wmedian(bt, ws), wmedian(tp, ws)
                        if tm - bm > 0.3:
                            z2, h2 = (bm + tm) / 2, tm - bm
                            b[2] = PED_ZMED * z2 + (1 - PED_ZMED) * b[2]
                            b[5] = PED_ZMED * h2 + (1 - PED_ZMED) * b[5]
                if c == 'pedestrian' and PED_MODE and PASS_GROUP is not None:
                    AUDIT['ped_clusters'] += 1
                    sel = bimodal_pick(np.flatnonzero(member), pid, boxes, scores)
                    if sel is not None:
                        AUDIT['ped_bimodal'] += 1
                        sw_ = scores[sel]
                        bs = (boxes[sel, :6] * sw_[:, None]).sum(0) / sw_.sum()
                        if PED_MODE == 1:
                            b[:6] = bs
                        else:
                            b[:2] = 0.5 * b[:2] + 0.5 * bs[:2]
                sc = float(min(w.sum() / n, 1.0))
                if c == 'pedestrian' and PED_AGREE and PASS_GROUP is not None:   # Codex idea 1: rank by geometric agreement
                    rb = rep_boxes(np.flatnonzero(member), pid, boxes, scores)
                    if len(rb) >= 3:
                        fb = np.array(b.tolist() + [yaw])
                        q = float(np.median([iou3d(fb, x[:7]) for x in rb]))
                        sc = sc * (1 - PED_AGREE * (1 - q))
                if c in ONE_VOTE:   # confidence counts only the best box of each (model, height) pass in the cluster; geometry unchanged
                    mi = np.flatnonzero(member)
                    best = {}
                    for j in mi:          # mi is in descending score order, so the first box seen per pass is its best
                        best.setdefault(pid[j], scores[j])
                    sc = float(min(sum(best.values()) / n, 1.0))
                if c == 'truck' and TRUCK_SCORE:   # experiment: rank fused truck boxes by a different aggregate (default: sum/passes)
                    k = float(TRUCK_SCORE.split(':')[1]) if ':' in TRUCK_SCORE else 1.0
                    sc = float(min(k * (w.max() if TRUCK_SCORE.startswith('max') else w.mean() if TRUCK_SCORE.startswith('mean')
                                         else np.sqrt(w.sum() * w.max())), 1.0))
                dets.append({'box': b.tolist() + [yaw], 'score': sc, 'class': c})
        out[f] = dets
    return out


def vkey(p):
    k = np.floor(p / V).astype(np.int64) + 2000
    return k[:, 0] * 4096 * 4096 + k[:, 1] * 4096 + k[:, 2]


def in_box(p, c, d, yaw):
    q = p - c
    cs, sn = np.cos(-yaw), np.sin(-yaw)
    x = q[:, 0] * cs - q[:, 1] * sn
    y = q[:, 0] * sn + q[:, 1] * cs
    return (np.abs(x) <= d[0] / 2) & (np.abs(y) <= d[1] / 2) & (np.abs(q[:, 2]) <= d[2] / 2)


def static_down(pred, clouds, exclude=()):
    dev = load(DEV)
    planes = np.array([fit_ground(dev[f], np.random.default_rng(i)) for i, f in enumerate(dev)])
    R, t = level_transform(np.median(planes, 0), GZ)

    def nonground(p):
        q = p @ R.T + t
        return q[q[:, 2] > GZ + GCUT]
    frames = [f for f in dev if f not in exclude]
    allk = np.concatenate([np.unique(vkey(nonground(dev[f]))) for f in frames])
    u, cnt = np.unique(allk, return_counts=True)
    S = u[cnt >= FRAC * len(frames)]
    out, changed = {}, 0
    for f, dets in pred.items():
        p = nonground(clouds[f])
        is_static = np.isin(vkey(p), S)
        new = []
        for d in dets:
            d = dict(d)
            if d['class'] == 'pedestrian':
                b = np.array(d['box'], float)
                h = R @ np.array([np.cos(b[6]), np.sin(b[6]), 0.0])
                m = in_box(p, R @ b[:3] + t, b[3:6], np.arctan2(h[1], h[0]))
                sf = float(is_static[m].mean()) if m.sum() else 0.0
                changed += sf > 0
                if os.environ.get('FUSE_KEEP_SF'):   # experiment only: remember the static fraction so a later step can undo/soften the penalty
                    d['sf'] = sf
                d['score'] = d['score'] * (1 - DOWN * sf)
            new.append(d)
        out[f] = new
    return out, {'static_voxels': int(len(S)), 'map_frames': len(frames), 'pedestrians_downweighted': int(changed)}


if __name__ == '__main__':
    clouds = load(sys.argv[1])
    ids = list(clouds)
    passes, names = [], []
    for arg in sys.argv[3:]:
        path, gs, keep = (arg.split(':') + ['', ''])[:3]
        data = json.loads(Path(path).read_text())
        for g in (gs.split(',') if gs else list(data)):
            p = data[g]
            assert set(p) == set(ids), (path, g)
            if keep:
                kc = set(keep.split(','))
                p = {f: [d for d in p[f] if d['class'] in kc] for f in p}
            passes.append(p)
            names.append(f'{Path(path).stem}@{g}{"#" + keep if keep else ""}')
    fz = fuse(passes, ids)
    final, info = static_down(fz, clouds, exclude=set(ids))
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final))
    info.update(passes=names, frames=len(ids), detections={c: sum(d['class'] == c for v in final.values() for d in v) for c in C})
    out.with_name(out.stem + '_info.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != 'passes'}), 'passes', len(names))
