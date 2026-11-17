"""Independent audit of CARLA native clips and their OpenPCDet conversion.

Yaw / handedness: semantic-LiDAR returns carry the hit actor ID. For every actor with >= 20 own returns we measure the fraction
of those returns inside the converted box (0.15 m margin) under the chosen convention AND under three wrong ones
(yaw sign flipped, no y flip on points, yaw + 90 deg). The chosen convention must be uniquely best and near 1.
Point-to-box alignment: own-return centroid offset in the box frame, and the share of own returns that fall outside the box.
Sensor/geometry: full azimuth coverage, vertical span, ground height, crop retention.
Labels: annotation rule re-derived, converted files byte-match a re-conversion, temporal ID continuity and speed plausibility.
"""
import argparse
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import convert_carla as C

SEM = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('cos', '<f4'), ('obj', '<u4'), ('tag', '<u4')])


def frac_inside(xyz, box, margin=0.15):
    return float(C.points_in_box(xyz, box, margin=margin).mean()) if len(xyz) else float('nan')


def local_offsets(xyz, box):
    d = xyz - box[:3]
    c, s = math.cos(-box[6]), math.sin(-box[6])
    return np.c_[c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]]


def load(p):
    return np.load(io.BytesIO(Path(p).read_bytes()))


def audit(native, converted=None, max_sem_frames=None):
    res = {'conventions': defaultdict(lambda: defaultdict(list)), 'per_blueprint_inside': defaultdict(list),
           'centroid_offset_norm': defaultdict(list), 'outside_share': defaultdict(list)}
    geo = defaultdict(list)
    tracks = defaultdict(list)
    problems = []
    n_sem = 0
    for cdir in sorted(Path(native).glob('clip_*')):
        meta = json.loads((cdir / 'clip.json').read_text())
        for k, f in enumerate(meta['frames']):
            rec = json.loads((cdir / 'actors' / f'{f:010d}.json').read_text())
            raw = load(cdir / 'lidar' / f'{f:010d}.npy')
            if rec['frame'] != f or rec['lidar_points'] != len(raw):
                problems.append(('frame/points mismatch', cdir.name, f))
            pts = C.native_points_rh(raw)
            az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
            geo['azimuth_bins_filled_of_360'].append(int(len(np.unique(np.floor(az).astype(int)))))
            r = np.linalg.norm(pts[:, :3], axis=1)
            el = np.degrees(np.arcsin(np.clip(pts[:, 2] / np.maximum(r, 1e-6), -1, 1)))
            geo['elevation_min'].append(float(el.min()))
            geo['elevation_max'].append(float(el.max()))
            near = pts[(np.hypot(pts[:, 0], pts[:, 1]) > 4) & (np.hypot(pts[:, 0], pts[:, 1]) < 20)]
            if len(near):
                geo['ground_z_p10_4_20m'].append(float(np.percentile(near[:, 2], 10)))
            geo['native_points'].append(len(raw))
            geo['cropped_points'].append(int(len(C.crop(pts))))
            for a in rec['actors']:
                if C.classify(a['blueprint']):
                    tracks[(cdir.name, a['id'])].append((rec['timestamp'], C.box_from_record(a, rec)[:3], a['blueprint']))
            semp = cdir / 'semantic' / f'{f:010d}.npy'
            if not semp.exists() or (max_sem_frames and n_sem >= max_sem_frames):
                continue
            n_sem += 1
            sem = load(semp)
            sxyz = np.c_[sem['x'], -sem['y'], sem['z']].astype(np.float64)
            sxyz_noflip = np.c_[sem['x'], sem['y'], sem['z']].astype(np.float64)
            for a in rec['actors']:
                cls = C.classify(a['blueprint'])
                if not cls:
                    continue
                m = sem['obj'] == a['id']
                if m.sum() < 20:
                    continue
                box = C.box_from_record(a, rec)
                variants = {'chosen': (sxyz[m], box),
                            'yaw_sign_flipped': (sxyz[m], np.r_[box[:6], -box[6]]),
                            'points_not_y_flipped': (sxyz_noflip[m], box),
                            'yaw_plus_90deg': (sxyz[m], np.r_[box[:6], C.wrap(box[6] + math.pi / 2)])}
                for name, (x, b) in variants.items():
                    res['conventions'][cls][name].append(frac_inside(x, b))
                res['per_blueprint_inside'][a['blueprint']].append(frac_inside(sxyz[m], box))
                off = local_offsets(sxyz[m], box)
                res['centroid_offset_norm'][cls].append(float(np.hypot(*(np.median(off[:, :2], 0) / (box[3:5] / 2)))))
                res['outside_share'][cls].append(1 - frac_inside(sxyz[m], box, margin=0.0))
    # temporal continuity / speed plausibility
    speeds = defaultdict(list)
    for (clip, aid), seq in tracks.items():
        seq.sort(key=lambda t: t[0])
        for (t0, p0, bp), (t1, p1, _) in zip(seq, seq[1:]):
            v = float(np.linalg.norm(p1[:2] - p0[:2]) / max(t1 - t0, 1e-6))
            speeds[C.classify(bp)].append(v)
            if v > (6.0 if C.classify(bp) == 'pedestrian' else 30.0):  # runners reach ~4.5 m/s
                problems.append(('implausible speed', clip, aid, round(v, 2)))
    summ = lambda v: {'n': len(v), 'median': float(np.median(v)), 'p05': float(np.percentile(v, 5)), 'p95': float(np.percentile(v, 95)),
                      'min': float(np.min(v)), 'max': float(np.max(v))} if len(v) else {'n': 0}  # noqa: E731
    out = {'semantic_frames_used': n_sem,
           'yaw_handedness_inside_fraction': {c: {k: summ(v) for k, v in d.items()} for c, d in res['conventions'].items()},
           'per_blueprint_inside_median': {k: [len(v), float(np.median(v))] for k, v in sorted(res['per_blueprint_inside'].items())},
           'own_return_centroid_offset_normalised': {k: summ(v) for k, v in res['centroid_offset_norm'].items()},
           'own_returns_outside_box_share': {k: summ(v) for k, v in res['outside_share'].items()},
           'geometry': {k: summ(v) for k, v in geo.items()},
           'speed_m_s': {k: summ(v) for k, v in speeds.items()},
           'problems': problems[:50], 'n_problems': len(problems)}
    chosen_ok = all(d['chosen']['median'] >= 0.95 and
                    all(d['chosen']['median'] > d[o]['median'] + 0.02 for o in d if o != 'chosen' and d[o]['n'])
                    for c, d in out['yaw_handedness_inside_fraction'].items() if c != 'pedestrian') and \
        out['yaw_handedness_inside_fraction'].get('pedestrian', {}).get('chosen', {'median': 1})['median'] >= 0.9
    out['checks'] = {'chosen_convention_best_and_tight': bool(chosen_ok),
                     'full_azimuth': bool(min(geo['azimuth_bins_filled_of_360']) >= 359),
                     'no_problems': not problems}
    if converted:
        out['converted'] = audit_converted(native, converted)
        out['checks'].update(out['converted']['checks'])
    out['all_passed'] = all(out['checks'].values())
    return out


def audit_converted(native, converted):
    converted = Path(converted)
    recs = json.loads((converted / 'frames.json').read_text())
    bad, cls_n, sizes, hits, rngs = [], defaultdict(int), defaultdict(list), defaultdict(list), defaultdict(list)
    byclip = {}
    for cdir in sorted(Path(native).glob('clip_*')):
        byclip[json.loads((cdir / 'clip.json').read_text())['clip']] = cdir
    for r in recs:
        cdir = byclip[r['layout_id']]
        f = r['carla_frame']
        raw = load(cdir / 'lidar' / f'{f:010d}.npy')
        rec = json.loads((cdir / 'actors' / f'{f:010d}.json').read_text())
        pts, gt, _ = C.convert_frame(raw, rec, r.get('synthetic_containers'), r.get('container_noise_seed'))
        got = np.load(converted / 'points' / f"{r['frame']}.npy")
        if got.dtype != np.float32 or got.shape != pts.shape or not np.array_equal(got, pts):
            bad.append(('points differ', r['frame']))
        if not (np.all(got[:, :3] >= C.PCR[:3]) and np.all(got[:, :3] <= C.PCR[3:]) and np.all(got[:, 3] == 0)):
            bad.append(('crop/intensity', r['frame']))
        lines = (converted / 'labels' / f"{r['frame']}.txt").read_text().splitlines()
        if len(lines) != len(gt) or r['ground_truth'] != gt:
            bad.append(('labels differ', r['frame']))
        for ln, g in zip(lines, gt):
            *nums, c = ln.split()
            if c != g['class'] or not np.allclose([float(x) for x in nums], g['box'], atol=0, rtol=0):
                bad.append(('label text', r['frame']))
            cls_n[c] += 1
            b = np.asarray(g['box'])
            sizes[c].append(b[3:6])
            rngs[c].append(float(math.hypot(b[0], b[1])))
        for d in r['object_details']:
            if d['annotated']:
                hits[d['class']].append(d['hits'])
            if d['annotated'] != (d['hits'] >= C.MIN_POINTS and d['corners_in_range'] >= C.MIN_CORNERS):
                bad.append(('annotation rule', r['frame'], d['id']))
    ids = (converted / 'ImageSets' / 'train.txt').read_text().split()
    return {'frames': len(recs), 'clips': len({r['layout_id'] for r in recs}), 'boxes': dict(cls_n),
            'frames_with_class': {c: sum(any(g['class'] == c for g in r['ground_truth']) for r in recs) for c in ('vehicle', 'pedestrian', 'truck')},
            'median_lwh': {c: np.median(v, 0).round(2).tolist() for c, v in sizes.items()},
            'median_range_m': {c: float(np.median(v)) for c, v in rngs.items()},
            'median_hits': {c: float(np.median(v)) for c, v in hits.items()},
            'points_median': float(np.median([r['points'] for r in recs])),
            'problems': bad[:30],
            'checks': {'reconversion_byte_identical_and_rules': not bad, 'imageset_matches': ids == [r['frame'] for r in recs],
                       'unique_ids': len(set(ids)) == len(ids)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--native', type=Path, required=True)
    ap.add_argument('--converted', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-sem-frames', type=int)
    a = ap.parse_args(argv)
    out = audit(a.native, a.converted, a.max_sem_frames)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({'checks': out['checks'], 'all_passed': out['all_passed'],
                      'yaw': {c: {k: round(v['median'], 3) for k, v in d.items()} for c, d in out['yaw_handedness_inside_fraction'].items()}},
                     indent=1))


if __name__ == '__main__':
    main()
