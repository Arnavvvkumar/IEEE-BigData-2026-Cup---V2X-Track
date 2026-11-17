"""Convert native CARLA clips (carla_gen.py) to the existing three-class OpenPCDet custom format.

Conventions (match data/codex_openpcdet_custom_v1 and the Generator V2 export):
  points/<id>.npy  float32 (N, 4) x, y, z, intensity in the RIGHT-HANDED sensor frame (x fwd, y left, z up), sensor at origin.
                   CARLA native LiDAR buffers are sensor-local left-handed (y right) -> y := -y.
                   Intensity is written as 0 because every canonical Generator V2 frame has intensity 0 (native kept untouched).
                   Points are cropped to the Track B POINT_CLOUD_RANGE [-40,-40,-8, 40,40,2].
  labels/<id>.txt  "x y z l w h yaw class" per line, box centre, yaw (rad, CCW from +x in the RH sensor frame).
  Annotation rule = Generator V2 rule: hits >= 3 and >= 2 box corners inside POINT_CLOUD_RANGE, where hits = native LiDAR
                   returns inside the box above a 0.10 m ground slab.
  Classes: walker.pedestrian.* -> pedestrian; TRUCKS (box truck, semi, fire truck, bus) -> truck; other 4-wheel -> vehicle.
                   This matches Generator V2's taxonomy (its trucks are semi/box_truck/bus, median length 9.7 m; vans are cars).
                   Blueprint IDs are kept in frames.json object_details so a different truck map can be re-derived.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PCR = np.array([-40.0, -40.0, -8.0, 40.0, 40.0, 2.0])
TRUCKS = {'vehicle.carlamotors.carlacola', 'vehicle.carlamotors.european_hgv', 'vehicle.carlamotors.firetruck',
          'vehicle.mitsubishi.fusorosa'}
MIN_POINTS, MIN_CORNERS, GROUND_SLAB = 3, 2, 0.10
REFINE_PATH = Path(__file__).with_name('bbox_refinement.json')
REFINE = {k: v for k, v in json.loads(REFINE_PATH.read_text())['blueprints'].items() if v['apply']} if REFINE_PATH.exists() else {}


def classify(bp):
    if bp.startswith('walker.pedestrian.'):
        return 'pedestrian'
    if bp.startswith('vehicle.'):
        return 'truck' if bp in TRUCKS else 'vehicle'
    return None


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_sensor_rh(points_world_lh, sensor_matrix):
    """World (CARLA, left-handed) -> sensor-local right-handed."""
    m = np.asarray(sensor_matrix, dtype=np.float64)
    inv = np.linalg.inv(m)
    p = np.c_[points_world_lh, np.ones(len(points_world_lh))] @ inv.T
    p = p[:, :3]
    p[:, 1] *= -1
    return p


def box_from_record(actor, sensor_rec, refine=True):
    """7-dof RH sensor-frame box from a carla_gen actor record (assumes a LEVEL sensor: roll = pitch = 0).
    With refine=True, rigid vehicle blueprints listed in bbox_refinement.json get their measured tight box."""
    roll, pitch, syaw = sensor_rec['sensor_transform_carla']['rotation']
    if abs(roll) > 1e-3 or abs(pitch) > 1e-3:
        raise ValueError('converter assumes a level sensor')
    c = to_sensor_rh(np.asarray([actor['center_world_carla']]), sensor_rec['sensor_matrix_carla'])[0]
    ex, ey, ez = actor['extent']
    yaw = wrap(-math.radians(actor['yaw_deg_carla'] - syaw))
    box = np.array([c[0], c[1], c[2], 2 * ex, 2 * ey, 2 * ez, yaw])
    r = REFINE.get(actor['blueprint']) if refine else None
    if r:
        ox, oy, oz = r['local_centre_offset']
        cy, sy = math.cos(yaw), math.sin(yaw)
        box[0] += cy * ox - sy * oy
        box[1] += sy * ox + cy * oy
        box[2] += oz
        box[3:6] = r['tight_lwh']
    return box


def points_in_box(xyz, box, margin=0.0, zmin_offset=0.0):
    d = xyz[:, :3] - box[:3]
    c, s = math.cos(-box[6]), math.sin(-box[6])
    lx = c * d[:, 0] - s * d[:, 1]
    ly = s * d[:, 0] + c * d[:, 1]
    return ((np.abs(lx) <= box[3] / 2 + margin) & (np.abs(ly) <= box[4] / 2 + margin) &
            (d[:, 2] >= -box[5] / 2 + zmin_offset - margin) & (d[:, 2] <= box[5] / 2 + margin))


def corners(box):
    x, y, z, l, w, h, yaw = box
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dx in (-l / 2, l / 2):
        for dy in (-w / 2, w / 2):
            for dz in (-h / 2, h / 2):
                out.append([x + c * dx - s * dy, y + s * dx + c * dy, z + dz])
    return np.asarray(out)


def corners_in_range(box):
    k = corners(box)
    return int(np.sum(np.all((k >= PCR[:3]) & (k <= PCR[3:]), axis=1)))


def native_points_rh(raw):
    p = np.asarray(raw, dtype=np.float32).copy()
    p[:, 1] *= -1
    return p


def crop(p):
    m = np.all((p[:, :3] >= PCR[:3]) & (p[:, :3] <= PCR[3:]), axis=1)
    return p[m]


def convert_frame(raw, rec, containers=None, noise_seed=None):
    """Returns (points float32 Nx4 with intensity 0, ground_truth list, object_details list).
    containers: optional list of synthetic container boxes (RH sensor frame) ray-cast into the native returns
    (containers.py); they are labelled 'truck' with the same annotation rule. noise_seed makes insertion deterministic."""
    pts = native_points_rh(raw)
    gt, details = [], []
    if containers:
        import containers as K
        rng = np.random.default_rng(noise_seed)
        for i, cb in enumerate(containers):
            cb = np.asarray(cb, dtype=float)
            pts, _ = K.insert(pts, cb, rng)
        for i, cb in enumerate(containers):
            cb = np.asarray(cb, dtype=float)
            hits = int(points_in_box(pts, cb, zmin_offset=GROUND_SLAB).sum())
            cir = corners_in_range(cb)
            ann = bool(hits >= MIN_POINTS and cir >= MIN_CORNERS)
            details.append({'id': f'container_{i}', 'class': 'truck', 'blueprint': 'synthetic.container', 'box': cb.tolist(),
                            'hits': hits, 'semantic_hits': 0, 'corners_in_range': cir, 'annotated': ann,
                            'range_m': float(math.hypot(cb[0], cb[1]))})
            if ann:
                gt.append({'box': cb.tolist(), 'class': 'truck'})
    for a in rec['actors']:
        cls = classify(a['blueprint'])
        if cls is None:
            continue
        box = box_from_record(a, rec)
        hits = int(points_in_box(pts, box, zmin_offset=GROUND_SLAB).sum())
        cir = corners_in_range(box)
        ann = bool(hits >= MIN_POINTS and cir >= MIN_CORNERS)
        details.append({'id': a['id'], 'class': cls, 'blueprint': a['blueprint'], 'box': box.tolist(), 'hits': hits,
                        'semantic_hits': int(rec.get('semantic_actor_hits', {}).get(str(a['id']), 0)),
                        'corners_in_range': cir, 'annotated': ann, 'range_m': float(math.hypot(box[0], box[1]))})
        if ann:
            gt.append({'box': box.tolist(), 'class': cls})
    out = crop(pts)
    out[:, 3] = 0.0
    return out.astype(np.float32), gt, details


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_exact(path):
    b = Path(path).read_bytes()
    if len(b) != Path(path).stat().st_size:
        raise IOError(f'short read {path}')
    return b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--native', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--containers-seed', type=int, help='insert 1-2 synthetic containers per clip (containers.py)')
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(a.out)
    for d in ('points', 'labels', 'ImageSets'):
        (a.out / d).mkdir(parents=True)
    records, ids = [], []
    import io
    for cdir in sorted(a.native.glob('clip_*')):
        meta = json.loads((cdir / 'clip.json').read_text())
        clip = meta['clip']
        conts = []
        if a.containers_seed is not None:
            import containers as K
            raws = [np.load(io.BytesIO(read_exact(cdir / 'lidar' / f'{f:010d}.npy'))) for f in meta['frames']]
            recs = [json.loads((cdir / 'actors' / f'{f:010d}.json').read_text()) for f in meta['frames']]
            boxes = [[box_from_record(x, r) for x in r['actors'] if classify(x['blueprint'])] for r in recs]
            crng = np.random.default_rng([a.containers_seed, clip])
            conts = [b.tolist() for b in K.place([native_points_rh(x) for x in raws], boxes, crng, int(crng.integers(1, 3)))]
        for k, f in enumerate(meta['frames']):
            raw = np.load(io.BytesIO(read_exact(cdir / 'lidar' / f'{f:010d}.npy')))
            rec = json.loads((cdir / 'actors' / f'{f:010d}.json').read_text())
            nseed = [a.containers_seed, clip, k] if conts else None
            pts, gt, det = convert_frame(raw, rec, conts, nseed)
            fid = f'carla_L{clip:04d}_F{k:02d}'
            np.save(a.out / 'points' / f'{fid}.npy', pts)
            (a.out / 'labels' / f'{fid}.txt').write_text(''.join(
                ' '.join(format(float(x), '.17g') for x in g['box']) + ' ' + g['class'] + '\n' for g in gt))
            cc = {c: sum(g['class'] == c for g in gt) for c in ('vehicle', 'pedestrian', 'truck')}
            records.append({'split': 'train', 'frame': fid, 'scenario': 'carla_town02_roadside', 'agent': '-1',
                            'layout_id': clip, 'layout_family': 'carla_town02', 'frame_in_layout': k,
                            'layout_fingerprint': hashlib.sha256(json.dumps(meta['site_navigation_point']).encode()).hexdigest()[:16],
                            'carla_frame': f, 'timestamp': rec['timestamp'],
                            'sensor': {'placement': 'carla_sidewalk', 'height_m': meta['mount_height_m'], 'roll_deg': 0.0,
                                       'pitch_deg': 0.0, 'yaw_deg_carla': meta['mount_transform_carla']['rotation'][2],
                                       'xyz_world_carla': meta['mount_transform_carla']['location']},
                            'measurement': {'profile': 'lidar64_level', 'returns_before_crop': int(len(raw)), 'points': int(len(pts))},
                            'points': int(len(pts)), 'objects': len(gt), 'class_counts': cc, 'ground_truth': gt,
                            'object_details': det, 'synthetic_containers': conts, 'container_noise_seed': nseed,
                            'native_lidar_sha256': sha(cdir / 'lidar' / f'{f:010d}.npy'),
                            'native_actors_sha256': sha(cdir / 'actors' / f'{f:010d}.json'),
                            'points_sha256': sha(a.out / 'points' / f'{fid}.npy'),
                            'labels_sha256': sha(a.out / 'labels' / f'{fid}.txt')})
            ids.append(fid)
    (a.out / 'ImageSets' / 'train.txt').write_text(''.join(i + '\n' for i in ids))
    (a.out / 'frames.json').write_text(json.dumps(records))
    prov = {'converter_sha256': sha(__file__), 'native_root': str(a.native), 'frames': len(ids),
            'clips': len({r['layout_id'] for r in records}), 'point_cloud_range': PCR.tolist(),
            'annotation_rule': {'min_points': MIN_POINTS, 'min_corners_in_range': MIN_CORNERS, 'ground_slab_m': GROUND_SLAB},
            'truck_blueprints': sorted(TRUCKS), 'intensity': 'zeroed to match canonical Generator V2 frames',
            'bbox_refinement_sha256': sha(REFINE_PATH) if REFINE_PATH.exists() else None,
            'refined_blueprints': sorted(REFINE),
            'containers_seed': a.containers_seed,
            'containers_sha256': sha(Path(__file__).with_name('containers.py')) if a.containers_seed is not None else None,
            'frames_json_sha256': sha(a.out / 'frames.json')}
    (a.out / 'provenance.json').write_text(json.dumps(prov, indent=1))
    print(json.dumps({k: prov[k] for k in ('frames', 'clips', 'frames_json_sha256')}))


if __name__ == '__main__':
    main()
