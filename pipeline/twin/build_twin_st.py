"""Site-twin training frames: sampled range-image background of the real fixed sensor + simulated objects + no-return origin points.
 * Background: range_bg model built from UNLABELLED real dev clouds (non-forbidden). Before fitting, points inside our own
   detector's boxes (synthetic-trained detector, no real labels) are removed so parked real objects do not become background.
 * Objects: points inside the synthetic boxes of the canonical synthetic set (Generator V2 / CARLA), moved to a random bearing
   around the sensor, dropped onto the twin's local ground, kept only where they are in free space; they occlude the background.
 * No-return beams: points at exactly (0,0,0) in the sensor frame, count drawn from the real per-scan counts.
Labels are the synthetic boxes only. No real label and no real object point enters the output.
Output (OpenPCDet custom layout, canonical frame, ground z = -3.0): points/ labels/ ImageSets/train.txt custom_infos_train.pkl,
plus sensor/<id>.npy (native sensor frame xyz incl. origin points) for the first --keep-sensor frames."""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import range_bg as T  # noqa: E402
sys.path.insert(0, str(HERE.parent / 'erroranalysis'))
from groundplane import level_transform  # noqa: E402

import os  # noqa: E402

GZ = -3.0
G = {}
# Defaults reproduce claude_twin_st_v1 (small trucks only). PEDVARY=1 PREFIX=twinpv SEED0=500000 gives the size-varied pedestrian set.
PEDVARY = os.environ.get('PEDVARY', '0') == '1'
LONGLOW = os.environ.get('LONGLOW', '0') == '1'  # LONGLOW=1 PREFIX=twinll SEED0=700000 gives the long-low truck set
PREFIX = os.environ.get('PREFIX', 'twinst')
SEED0 = int(os.environ.get('SEED0', '300000'))


def in_box(p, b, grow=0.0):
    q = p[:, :3] - b[:3]
    cs, sn = np.cos(-b[6]), np.sin(-b[6])
    x = q[:, 0] * cs - q[:, 1] * sn
    y = q[:, 0] * sn + q[:, 1] * cs
    return (np.abs(x) <= b[3] / 2 + grow) & (np.abs(y) <= b[4] / 2 + grow) & (np.abs(q[:, 2]) <= b[5] / 2 + grow)


def setup(args):
    preds = json.loads(Path(args.pred).read_text())
    orig = dict(T.clouds)
    for f in T.ids:
        p = orig[f]
        p = p[np.linalg.norm(p, axis=1) > 0]
        drop = np.zeros(len(p), bool)
        for d in preds[f]:
            if d['score'] >= 0.25:
                drop |= in_box(p, np.array(d['box'], float), grow=0.25)
        T.clouds[f] = p[~drop]
    model = T.build(T.ids)
    T.clouds.update(orig)
    R, t = level_transform(np.load(HERE / 'median_plane.npy'), GZ)
    RB = np.full(T.NE * T.NA, np.inf)
    RB[model['cell']] = model['range']
    pc = (model['dir'] * model['range'][:, None]) @ R.T + t
    low = pc[pc[:, 2] < GZ + 0.8]
    key = (np.floor(low[:, 0]).astype(int) + 200) * 400 + (np.floor(low[:, 1]).astype(int) + 200)
    order = np.argsort(key)
    key, z = key[order], low[order, 2]
    st = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    H = np.full(400 * 400, np.nan)
    for s, e in zip(st, np.r_[st[1:], len(key)]):
        if e - s >= 3:
            H[key[s]] = np.percentile(z[s:e], 10)
    zeros = np.array([int((np.linalg.norm(orig[f], axis=1) == 0).sum()) for f in T.ids])
    src = Path(args.src)
    G.update(model=model, R=R, t=t, RB=RB, H=H, zeros=zeros, src=src, out=Path(args.out), keep=args.keep_sensor,
             ids=(src / 'ImageSets/train.txt').read_text().split())
    print('model cells', len(model['cell']), 'ground cells', int(np.isfinite(H).sum()), 'zeros median', int(np.median(zeros)), flush=True)


def cells_of(pS):
    r = np.linalg.norm(pS, axis=1)
    el = np.degrees(np.arcsin(np.clip(pS[:, 2] / np.maximum(r, 1e-6), -1, 1)))
    az = np.arctan2(pS[:, 1], pS[:, 0])
    ie = np.clip(np.floor((el - T.E0) / T.DE).astype(int), 0, T.NE - 1)
    ia = np.floor((az + np.pi) / (2 * np.pi) * T.NA).astype(int) % T.NA
    return ie, ia, r


def work(i):
    rng = np.random.default_rng(SEED0 + i)
    m, R, t, RB, H = G['model'], G['R'], G['t'], G['RB'], G['H']
    keep = rng.random(len(m['prob'])) < m['prob']
    d = m['dir'][keep] + rng.normal(0, 0.0015, (keep.sum(), 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    br = m['range'][keep] + rng.normal(0, 1, keep.sum()) * np.clip(m['sigma'][keep], 0.01, 0.2)
    bcell = m['cell'][keep]
    bg = d * br[:, None]
    OB = np.full(T.NE * T.NA, np.inf)
    objs, boxes, names = [], [], []
    target = int(rng.integers(14, 36))
    for f in rng.choice(G['ids'], 4, replace=False):
        if len(boxes) >= target:
            break
        pts = np.load(G['src'] / 'points' / f'{f}.npy')[:, :3].astype(np.float64)
        pts = pts[pts[:, 2] > GZ + 0.08]
        lines = (G['src'] / 'labels' / f'{f}.txt').read_text().splitlines()
        lines.sort(key=lambda s: s.split()[-1] != 'truck')  # trucks first so they are well represented
        for ln in lines:
            *n, c = ln.split()
            b = np.array([float(x) for x in n])
            o = pts[in_box(pts, b, grow=0.05)]
            if len(o) < 5 or len(boxes) >= target:
                continue
            new = None
            if c == 'truck' and LONGLOW:
                # Long, low truck-class objects (trailer / container / low bus shape, ~8-10 m long but only ~2.5 m high). The fused
                # detections and the point clusters on the unlabelled real scans show several such objects (cluster extent about
                # 8-9 x 3 x 2.3 m) that the detectors score low; only 6-8% of bank trucks have this shape (they are ~3.6 m high).
                r = rng.random()
                if r < 0.55:
                    new = rng.uniform(7.3, 10.8), rng.uniform(2.55, 3.2), rng.uniform(2.25, 2.85)
                elif r < 0.75:
                    new = rng.uniform(4.6, 6.8), rng.uniform(1.9, 2.45), rng.uniform(1.8, 2.5)
            elif c == 'truck' and rng.random() < 0.85:
                # Resize the synthetic truck to a pickup / van / box-truck size (real trucks are ~5-6.5 m, bank trucks ~8.5 m).
                # Points are scaled in the box frame (length about the centre, height about the box bottom); labels follow.
                new = rng.uniform(4.6, 6.8), rng.uniform(1.9, 2.45), rng.uniform(1.8, 2.5)
            elif c == 'pedestrian' and PEDVARY and rng.random() < 0.85:
                # Vary the pedestrian footprint and height (bank pedestrians have one fixed box size, so detectors learn a constant
                # size); the box stays tight to the scaled body, so the model has to read the size from the points.
                s = rng.uniform(0.5, 1.0)
                new = s * rng.uniform(0.9, 1.2), s * rng.uniform(0.85, 1.1), rng.uniform(1.5, 1.95)
            if new is not None:
                nl, nw, nh = new
                sl, sw, sh = nl / b[3], nw / b[4], nh / b[5]
                cs0, sn0 = np.cos(-b[6]), np.sin(-b[6])
                q = o - b[:3]
                lx = q[:, 0] * cs0 - q[:, 1] * sn0
                ly = q[:, 0] * sn0 + q[:, 1] * cs0
                zb = -b[5] / 2
                lz = zb + (q[:, 2] - zb) * sh
                lx, ly = lx * sl, ly * sw
                cs1, sn1 = np.cos(b[6]), np.sin(b[6])
                nb0 = b.copy()
                nb0[3:6] = nl, nw, nh
                # centre z: bottom stays at b[2]-b[5]/2, new centre = bottom + nh/2
                nb0[2] = (b[2] - b[5] / 2) + nh / 2
                o = np.stack([lx * cs1 - ly * sn1 + b[0], lx * sn1 + ly * cs1 + b[1], lz + b[2]], 1)
                b = nb0
            for _ in range(6):
                th = rng.uniform(0, 2 * np.pi)
                cs, sn = np.cos(th), np.sin(th)
                rot = np.array([[cs, -sn], [sn, cs]])
                c2 = rot @ (b[:2] - t[:2]) + t[:2]
                hk = (int(np.floor(c2[0])) + 200) * 400 + (int(np.floor(c2[1])) + 200)
                if not (0 <= hk < H.size) or not np.isfinite(H[hk]) or abs(H[hk] - GZ) > 1.0:
                    continue
                if any(np.hypot(*(c2 - bb[:2])) < (np.hypot(b[3], b[4]) + np.hypot(bb[3], bb[4])) / 2 for bb in boxes):
                    continue
                dz = H[hk] - GZ
                oc = o.copy()
                oc[:, :2] = (o[:, :2] - t[:2]) @ rot.T + t[:2]
                oc[:, 2] += dz
                oS = (oc - t) @ R
                ie, ia, r = cells_of(oS)
                if np.mean(r < RB[ie * T.NA + ia] - 0.3) < 0.9:
                    continue
                nb = b.copy()
                nb[:2], nb[2], nb[6] = c2, b[2] + dz, b[6] + th
                for de in (-2, -1, 0, 1, 2):
                    for da in (-1, 0, 1):
                        np.minimum.at(OB, np.clip(ie + de, 0, T.NE - 1) * T.NA + (ia + da) % T.NA, r)
                objs.append(oS)
                boxes.append(nb)
                names.append(c)
                break
    vis = br <= OB[bcell]
    k = int(rng.choice(G['zeros']))
    S = np.concatenate([bg[vis]] + objs + [np.zeros((k, 3))]).astype(np.float32)
    C = np.zeros((len(S), 4), np.float32)
    C[:, :3] = S.astype(np.float64) @ R.T + t
    fid = f'{PREFIX}_{i:06d}'
    np.save(G['out'] / 'points' / f'{fid}.npy', C)
    if i < G['keep']:
        np.save(G['out'] / 'sensor' / f'{fid}.npy', S)
    (G['out'] / 'labels' / f'{fid}.txt').write_text(''.join(' '.join(format(float(x), '.17g') for x in b) + ' ' + c + '\n' for b, c in zip(boxes, names)))
    return fid, np.array(names), np.array(boxes, dtype=np.float32).reshape(-1, 7), len(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)   # canonical synthetic dataset (object bank)
    ap.add_argument('--pred', required=True)  # our detector's predictions on the dev frames (sensor frame), to mask real objects
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--keep-sensor', type=int, default=400)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    for dname in ('points', 'labels', 'ImageSets', 'sensor'):
        (out / dname).mkdir(parents=True)
    setup(a)
    with Pool(a.workers) as pool:
        res = pool.map(work, range(a.n), chunksize=8)
    infos = [{'point_cloud': {'num_features': 4, 'lidar_idx': f}, 'annos': {'name': n, 'gt_boxes_lidar': b}} for f, n, b, _ in res]
    (out / 'ImageSets/train.txt').write_text('\n'.join(f for f, *_ in res) + '\n')
    (out / 'custom_infos_train.pkl').write_bytes(pickle.dumps(infos, protocol=4))
    allc = np.concatenate([n for _, n, _, _ in res]) if res else np.array([])
    man = {'frames': len(res), 'boxes': {c: int((allc == c).sum()) for c in ('vehicle', 'pedestrian', 'truck')},
           'boxes_per_frame_median': float(np.median([len(n) for _, n, _, _ in res])),
           'points_per_frame_median': int(np.median([k for *_, k in res])), 'object_bank': a.src,
           'real_data_use': 'unlabelled real dev clouds (non-forbidden) for the background range model and no-return counts; no real labels'}
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
