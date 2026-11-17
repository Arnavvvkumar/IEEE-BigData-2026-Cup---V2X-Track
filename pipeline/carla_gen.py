"""CPU-only CARLA 0.9.16 Town02 roadside capture with the retained 64-channel LEVEL profile.

Server: stock Windows CARLA started with -nullrhi (no GPU device) and world no_rendering_mode=True.
Each clip: a new roadside site (navigation-mesh sidewalk point + 3.2 m, level, random yaw), background 4-wheel traffic on
Traffic Manager autopilot, 1-3 trucks biased to the site, 6-14 AI walkers with varied blueprints and speeds.
Native output is preserved untouched (CARLA left-handed buffers + world actor records); conversion is a separate step.

Sensor profile = experiments/roadside_pilot/profiles/collector_50.yaml roadside_lidar (lidar64_level), attribute-for-attribute.
A co-located semantic LiDAR (same rays, no noise/drop) supplies per-point actor IDs for the alignment audit only.
"""
import argparse
import json
import math
import queue
import sys
import time
from pathlib import Path

import carla
import numpy as np

PROFILE = {  # collector_50.yaml -> actors[0].sensors[0] (64 level)
    'range': 120, 'channels': 64, 'rotation_frequency': 10, 'points_per_second': 655360,
    'lower_fov': -22.5, 'upper_fov': 22.5, 'horizontal_fov': 360, 'noise_stddev': 0.02, 'dropoff_general_rate': 0.05}
MOUNT_HEIGHT = 3.2
WORLD = {'fixed_delta_seconds': 0.1, 'substepping': True, 'max_substep_delta_time': 0.01, 'max_substeps': 16}
TRUCKS = ('vehicle.carlamotors.carlacola', 'vehicle.carlamotors.european_hgv', 'vehicle.carlamotors.firetruck',
          'vehicle.mitsubishi.fusorosa')
SEM_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('cos', '<f4'), ('obj', '<u4'), ('tag', '<u4')])


def xy(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def lidar_bp(lib, semantic=False):
    bp = lib.find('sensor.lidar.ray_cast_semantic' if semantic else 'sensor.lidar.ray_cast')
    for k, v in PROFILE.items():
        if semantic and k in ('noise_stddev', 'dropoff_general_rate'):
            continue
        bp.set_attribute(k, str(v))
    return bp


def actor_record(a, snap):
    t = snap.find(a.id).get_transform()
    bb = a.bounding_box
    c = t.transform(bb.location)
    return {'id': a.id, 'blueprint': a.type_id, 'wheels': a.attributes.get('number_of_wheels'),
            'center_world_carla': [c.x, c.y, c.z], 'extent': [bb.extent.x, bb.extent.y, bb.extent.z],
            'yaw_deg_carla': t.rotation.yaw + bb.rotation.yaw, 'pitch_deg': t.rotation.pitch, 'roll_deg': t.rotation.roll,
            'velocity': [v for v in (lambda s: (s.x, s.y, s.z))(snap.find(a.id).get_velocity())]}


class Gen:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(120)
        self.world = self.client.get_world()
        if not self.world.get_map().name.endswith('Town02'):
            self.world = self.client.load_world('Town02')
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.no_rendering_mode = not args.world_rendering
        for k, v in WORLD.items():
            setattr(s, k, v)
        self.world.apply_settings(s)
        self.tm = self.client.get_trafficmanager(args.tm_port)
        self.tm.set_synchronous_mode(True)
        self.lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawns = self.map.get_spawn_points()
        self.clear_world()

    def clear_world(self):
        acts = [a for a in self.world.get_actors() if a.type_id.startswith(('vehicle.', 'walker.', 'controller.', 'sensor.'))]
        for a in acts:
            if a.type_id.startswith('controller.'):
                a.stop()
        self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in acts], True)
        for _ in range(3):
            self.world.tick()

    def pick_site(self, rng, previous):
        for _ in range(2000):
            p = self.world.get_random_location_from_navigation()
            if p is None:
                continue
            near = sum(10 < xy(p, sp.location) < 38 for sp in self.spawns)
            if near < 3 or any(xy(p, q) < self.args.site_gap for q in previous):
                continue
            return p, near
        raise RuntimeError('no admissible site')

    def run_clip(self, idx, seed, previous, out):
        rng = np.random.default_rng(seed)
        self.world.set_pedestrians_seed(int(seed))
        self.tm.set_random_device_seed(int(seed))
        site, near = self.pick_site(rng, previous)
        yaw = float(rng.uniform(-180, 180))
        mount = carla.Transform(carla.Location(site.x, site.y, site.z + MOUNT_HEIGHT), carla.Rotation(yaw=yaw))
        spawned, controllers = [], []
        # trucks: 1-3, nearest spawn points to a random target distance in 10-40 m
        n_trucks = int(rng.integers(1, 4))
        free = list(self.spawns)
        truck_types = [TRUCKS[(idx + k) % len(TRUCKS)] for k in range(n_trucks)]
        for tt in truck_types:
            d = rng.uniform(10, 40)
            free.sort(key=lambda sp: abs(xy(sp.location, site) - d))
            for sp in free[:12]:
                a = self.world.try_spawn_actor(self.lib.find(tt), sp)
                if a:
                    spawned.append(a)
                    free.remove(sp)
                    break
        # background 4-wheel non-truck traffic; half biased near the site
        cars = sorted([bp for bp in self.lib.filter('vehicle.*') if bp.get_attribute('number_of_wheels').as_int() == 4
                       and bp.id not in TRUCKS], key=lambda b: b.id)
        n_cars = int(rng.integers(15, 31))
        near_sp = [sp for sp in free if xy(sp.location, site) < 60]
        order = list(rng.permutation(len(near_sp)))[: n_cars // 2]
        pool = [near_sp[i] for i in order] + [free[i] for i in rng.permutation(len(free))]
        used = set()
        for sp in pool:
            if sum(a.type_id not in TRUCKS for a in spawned) >= n_cars:
                break
            key = (round(sp.location.x, 1), round(sp.location.y, 1))
            if key in used:
                continue
            used.add(key)
            bp = cars[int(rng.integers(len(cars)))]
            if bp.has_attribute('color'):
                vals = bp.get_attribute('color').recommended_values
                bp.set_attribute('color', vals[int(rng.integers(len(vals)))])
            a = self.world.try_spawn_actor(bp, sp)
            if a:
                spawned.append(a)
        for a in spawned:
            a.set_autopilot(True, self.args.tm_port)
            self.tm.vehicle_percentage_speed_difference(a, float(rng.uniform(-20, 30)))
        # walkers: 6-14, varied blueprints, within 4-45 m of the site
        wbps = sorted(self.lib.filter('walker.pedestrian.*'), key=lambda b: b.id)
        n_walk = int(rng.integers(6, 15))
        cands = [p for p in (self.world.get_random_location_from_navigation() for _ in range(600))
                 if p and 4 < xy(p, site) < 45]
        rng.shuffle(cands)
        walkers, speeds = [], []
        for p in cands:
            if len(walkers) >= n_walk:
                break
            bp = wbps[int(rng.integers(len(wbps)))]
            if self.args.walker_vulnerable and bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            sp = bp.get_attribute('speed').recommended_values if bp.has_attribute('speed') else ['1.3', '2.5']
            run = rng.random() < 0.15
            speeds.append(float(sp[2] if run and len(sp) > 2 else sp[1]) * float(rng.uniform(0.8, 1.15)))
            a = self.world.try_spawn_actor(bp, carla.Transform(p + carla.Location(z=1.0), carla.Rotation(yaw=float(rng.uniform(0, 360)))))
            if a:
                walkers.append(a)
            else:
                speeds.pop()
        self.world.tick()
        if self.args.walker_mode == 'ai':
            cbp = self.lib.find('controller.ai.walker')
            for w in walkers:
                controllers.append(self.world.spawn_actor(cbp, carla.Transform(), attach_to=w))
            self.world.set_pedestrians_cross_factor(0.3)
            self.world.tick()
            for c, v in zip(controllers, speeds):
                c.start()
                c.go_to_location(self.world.get_random_location_from_navigation())
                c.set_max_speed(v)
        else:  # direct WalkerControl (AI-controller walkers return no LiDAR points in this CARLA build)
            for i, v in enumerate(speeds):
                if rng.random() < 0.15:
                    speeds[i] = 0.0  # standing pedestrians
            walk_state = [[float(rng.uniform(0, 2 * math.pi)), 0] for _ in walkers]
        # sensors
        lid = self.world.spawn_actor(lidar_bp(self.lib), mount)
        sem = self.world.spawn_actor(lidar_bp(self.lib, True), mount)
        ql, qs = queue.Queue(), queue.Queue()
        lid.listen(ql.put)
        sem.listen(qs.put)
        cdir = out / f'clip_{idx:03d}'
        for sub in ('lidar', 'semantic', 'actors'):
            (cdir / sub).mkdir(parents=True)
        tracked = spawned + walkers
        frames = []
        t0 = time.time()
        try:
            total = self.args.warmup + self.args.frames * self.args.step
            for k in range(total):
                if self.args.walker_mode == 'direct':
                    for w, st, v in zip(walkers, walk_state, speeds):
                        if k >= st[1]:  # new heading every 1.5-5 s, mostly gentle turns
                            st[0] += float(rng.normal(0, 0.7))
                            st[1] = k + int(rng.integers(15, 50))
                            w.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=math.cos(st[0]), y=math.sin(st[0])),
                                                                speed=v))
                f = self.world.tick()
                dl = self._get(ql, f)
                ds = self._get(qs, f)
                capture = k >= self.args.warmup and (k - self.args.warmup) % self.args.step == 0
                if not capture:
                    continue
                snap = self.world.get_snapshot()
                if snap.frame != f or dl.frame != f or ds.frame != f:
                    raise RuntimeError(f'frame mismatch {f} {snap.frame} {dl.frame} {ds.frame}')
                raw = np.frombuffer(dl.raw_data, dtype='<f4').reshape(-1, 4).copy()
                np.save(cdir / 'lidar' / f'{f:010d}.npy', raw)
                semraw = np.frombuffer(ds.raw_data, dtype=SEM_DTYPE).copy()
                if len(frames) % self.args.sem_every == 0:
                    np.save(cdir / 'semantic' / f'{f:010d}.npy', semraw)
                alive = [a for a in tracked if snap.find(a.id) is not None]
                st = lid.get_transform()
                rec = {'frame': f, 'timestamp': snap.timestamp.elapsed_seconds,
                       'sensor_transform_carla': {'location': [st.location.x, st.location.y, st.location.z],
                                                  'rotation': [st.rotation.roll, st.rotation.pitch, st.rotation.yaw]},
                       'sensor_matrix_carla': st.get_matrix(),
                       'lidar_points': int(len(raw)), 'semantic_points': int(len(semraw)),
                       'semantic_actor_hits': {str(i): int(n) for i, n in zip(*np.unique(semraw['obj'], return_counts=True))
                                               if any(a.id == int(i) for a in alive)},
                       'actors': [actor_record(a, snap) for a in alive]}
                (cdir / 'actors' / f'{f:010d}.json').write_text(json.dumps(rec))
                frames.append(f)
        finally:
            for s in (lid, sem):
                s.stop()
            for c in controllers:
                c.stop()
            self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in [lid, sem] + controllers + walkers + spawned], True)
            for _ in range(3):
                self.world.tick()
        meta = {'clip': idx, 'seed': int(seed), 'map': self.map.name, 'carla_server_version': self.client.get_server_version(),
                'carla_client_version': self.client.get_client_version(), 'site_navigation_point': [site.x, site.y, site.z],
                'spawn_points_10_38m': near, 'mount_transform_carla': {'location': [mount.location.x, mount.location.y, mount.location.z],
                                                                       'rotation': [0.0, 0.0, yaw]},
                'mount_height_m': MOUNT_HEIGHT, 'profile': PROFILE, 'world_settings': {**WORLD, 'synchronous_mode': True, 'no_rendering_mode': not self.args.world_rendering},
                'warmup_ticks': self.args.warmup, 'step_ticks': self.args.step, 'frames': frames,
                'trucks': [a.type_id for a in spawned if a.type_id in TRUCKS],
                'cars': sorted(a.type_id for a in spawned if a.type_id not in TRUCKS),
                'walkers': [w.type_id for w in walkers], 'walker_speeds': speeds,
                'walker_mode': self.args.walker_mode, 'walker_is_invincible': not self.args.walker_vulnerable,
                'semantic_saved_every': self.args.sem_every, 'wall_seconds': time.time() - t0}
        (cdir / 'clip.json').write_text(json.dumps(meta, indent=1))
        return site, meta

    @staticmethod
    def _get(q, f):
        while True:
            d = q.get(timeout=30)
            if d.frame == f:
                return d
            if d.frame > f:
                raise RuntimeError(f'sensor skipped frame {f}: got {d.frame}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--clips', type=int, default=25)
    ap.add_argument('--first-clip', type=int, default=0)
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--step', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--sem-every', type=int, default=4)
    ap.add_argument('--site-gap', type=float, default=15.0)
    ap.add_argument('--seed', type=int, default=88000)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2200)
    ap.add_argument('--tm-port', type=int, default=8210)
    ap.add_argument('--world-rendering', action='store_true', help='diagnostic: leave no_rendering_mode off')
    ap.add_argument('--walker-mode', choices=('ai', 'direct'), default='direct')
    ap.add_argument('--walker-vulnerable', action='store_true', help='diagnostic: is_invincible=false')
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    g = Gen(a)
    sites_file = a.out / 'sites.json'
    sites = json.loads(sites_file.read_text()) if sites_file.exists() else []
    previous = [carla.Location(*s['site']) for s in sites]
    for idx in range(a.first_clip, a.first_clip + a.clips):
        if (a.out / f'clip_{idx:03d}' / 'clip.json').exists():
            continue
        if (a.out / f'clip_{idx:03d}').exists():
            raise RuntimeError(f'partial clip_{idx:03d} exists; rename it before resuming')
        site, meta = g.run_clip(idx, a.seed + idx, previous, a.out)
        previous.append(site)
        sites.append({'clip': idx, 'site': [site.x, site.y, site.z], 'seed': meta['seed']})
        sites_file.write_text(json.dumps(sites, indent=1))
        print(json.dumps({'clip': idx, 'frames': len(meta['frames']), 'trucks': meta['trucks'], 'walkers': len(meta['walkers']),
                          'cars': len(meta['cars']), 'wall_s': round(meta['wall_seconds'], 1)}), flush=True)
    s = g.world.get_settings()
    s.synchronous_mode = False
    g.world.apply_settings(s)
    g.tm.set_synchronous_mode(False)


if __name__ == '__main__':
    sys.exit(main())
