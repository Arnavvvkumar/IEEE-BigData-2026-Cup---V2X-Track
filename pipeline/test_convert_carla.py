"""Synthetic-fixture tests for convert_carla.py (no simulator, no real data)."""
import math
import unittest

import numpy as np

import convert_carla as C


def carla_matrix(x, y, z, yaw_deg):
    """carla.Transform(Location(x,y,z), Rotation(yaw)).get_matrix() for roll = pitch = 0 (left-handed, z up)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]


def surface_points_world(center, extent, yaw_deg, n=400, seed=0):
    rng = np.random.default_rng(seed)
    loc = rng.uniform(-1, 1, (n, 3)) * extent
    face = rng.integers(0, 3, n)
    loc[np.arange(n), face] = np.sign(loc[np.arange(n), face]) * np.asarray(extent)[face]
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    w = np.c_[c * loc[:, 0] - s * loc[:, 1], s * loc[:, 0] + c * loc[:, 1], loc[:, 2]]
    return w + center


def to_sensor_lh(pw, m):
    inv = np.linalg.inv(np.asarray(m))
    return (np.c_[pw, np.ones(len(pw))] @ inv.T)[:, :3]


class Conventions(unittest.TestCase):
    def setUp(self):
        self.m = carla_matrix(100.0, -50.0, 3.4, 33.0)
        self.rec = {'sensor_transform_carla': {'location': [100, -50, 3.4], 'rotation': [0.0, 0.0, 33.0]}, 'sensor_matrix_carla': self.m}
        self.actor = {'id': 7, 'blueprint': 'vehicle.audi.a2', 'center_world_carla': [112.0, -41.0, 0.8],
                      'extent': [2.2, 0.9, 0.75], 'yaw_deg_carla': 101.0}

    def test_points_fall_inside_converted_box(self):
        pw = surface_points_world(np.array(self.actor['center_world_carla']), self.actor['extent'], self.actor['yaw_deg_carla'])
        raw = np.c_[to_sensor_lh(pw, self.m), np.ones(len(pw))].astype(np.float32)
        pts = C.native_points_rh(raw)
        box = C.box_from_record(self.actor, self.rec)
        self.assertTrue(C.points_in_box(pts, box, margin=1e-3).all())
        wrong = np.r_[box[:6], -box[6]]
        self.assertLess(C.points_in_box(pts, wrong, margin=1e-3).mean(), 0.8)
        self.assertLess(C.points_in_box(raw, box, margin=1e-3).mean(), 0.5)  # no y flip -> wrong
        self.assertAlmostEqual(box[3], 4.4)
        self.assertAlmostEqual(box[5], 1.5)

    def test_yaw_value(self):
        box = C.box_from_record(self.actor, self.rec)
        self.assertAlmostEqual(box[6], C.wrap(-math.radians(101 - 33)))

    def test_rejects_tilted_sensor(self):
        rec = dict(self.rec, sensor_transform_carla={'location': [0, 0, 0], 'rotation': [0.0, 1.6, 0.0]})
        with self.assertRaises(ValueError):
            C.box_from_record(self.actor, rec)

    def test_refinement_only_for_listed_blueprints(self):
        saved = dict(C.REFINE)
        try:
            C.REFINE.clear()
            C.REFINE['vehicle.audi.a2'] = {'local_centre_offset': [0.5, 0.0, 0.1], 'tight_lwh': [5.0, 2.0, 1.7]}
            b = C.box_from_record(self.actor, self.rec)
            b0 = C.box_from_record(self.actor, self.rec, refine=False)
            self.assertTrue(np.allclose(b[3:6], [5.0, 2.0, 1.7]))
            self.assertAlmostEqual(b[0] - b0[0], 0.5 * math.cos(b0[6]))
            self.assertAlmostEqual(b[1] - b0[1], 0.5 * math.sin(b0[6]))
            other = dict(self.actor, blueprint='vehicle.tesla.model3')
            self.assertTrue(np.allclose(C.box_from_record(other, self.rec), b0))
        finally:
            C.REFINE.clear()
            C.REFINE.update(saved)


class Frame(unittest.TestCase):
    def test_crop_intensity_rule_and_classes(self):
        m = carla_matrix(0, 0, 3.2, 0)
        rec = {'sensor_transform_carla': {'location': [0, 0, 3.2], 'rotation': [0, 0, 0]}, 'sensor_matrix_carla': m,
               'actors': [{'id': 1, 'blueprint': 'walker.pedestrian.0001', 'center_world_carla': [10, 5, 0.9], 'extent': [0.2, 0.2, 0.9], 'yaw_deg_carla': 0},
                          {'id': 2, 'blueprint': 'vehicle.mitsubishi.fusorosa', 'center_world_carla': [-20, 0, 1.6], 'extent': [5, 1.4, 1.6], 'yaw_deg_carla': 0},
                          {'id': 3, 'blueprint': 'vehicle.audi.a2', 'center_world_carla': [60, 0, 0.8], 'extent': [2, 1, 0.8], 'yaw_deg_carla': 0},
                          {'id': 4, 'blueprint': 'static.prop.bench', 'center_world_carla': [5, 5, 0.5], 'extent': [1, 1, 1], 'yaw_deg_carla': 0}]}
        # 3 returns on the walker (sensor LH: y = +5 world), 2 on the bus, 5 on the far car, and two out-of-range points
        lh = np.array([[10, 5, -2.0], [10, 5, -1.8], [10.05, 5, -2.2], [-20, 0, -2.0], [-19, 0.3, -1.5],
                       *[[60, 0, -2.5]] * 5, [0, 0, 5.0], [50, 0, -3.0]], dtype=np.float32)
        raw = np.c_[lh, np.full(len(lh), 0.7, np.float32)]
        pts, gt, det = C.convert_frame(raw, rec)
        self.assertTrue((pts[:, 3] == 0).all())
        self.assertTrue((np.abs(pts[:, :2]) <= 40).all() and (pts[:, 2] <= 2).all())
        self.assertEqual(len(pts), len(raw) - 5 - 2)
        by = {d['id']: d for d in det}
        self.assertNotIn(4, by)
        self.assertEqual(by[1]['class'], 'pedestrian')
        self.assertTrue(by[1]['annotated'])
        self.assertEqual(by[2]['class'], 'truck')
        self.assertFalse(by[2]['annotated'])  # 2 hits < 3
        self.assertEqual(by[3]['corners_in_range'], 0)
        self.assertFalse(by[3]['annotated'])  # out of range despite 5 hits
        self.assertEqual([g['class'] for g in gt], ['pedestrian'])
        self.assertAlmostEqual(gt[0]['box'][1], -5.0, places=5)  # world/LH y=+5 -> RH y=-5


class Containers(unittest.TestCase):
    def test_ray_insertion_occludes_and_lands_on_surface(self):
        import containers as K
        box = np.array([20.0, 0.0, -2.0, 8.0, 3.0, 2.5, 0.3])
        rng = np.random.default_rng(0)
        # rays: one straight through the box to a far wall, one past the box, one ending before the box
        pts = np.array([[40.0, 0.0, -2.0, 0], [40.0, 30.0, -2.0, 0], [10.0, 0.0, -2.0, 0]], dtype=np.float32)
        new, n = K.insert(pts, box, rng)
        self.assertEqual(n, 1)
        self.assertTrue(K.inside(new[:1], box, margin=0.1).all())
        self.assertLess(np.linalg.norm(new[0, :3]), 20.0)
        self.assertTrue(np.array_equal(new[1:], pts[1:]))

    def test_container_labelled_truck_and_reproducible(self):
        m = carla_matrix(0, 0, 3.2, 0)
        rec = {'sensor_transform_carla': {'location': [0, 0, 3.2], 'rotation': [0, 0, 0]}, 'sensor_matrix_carla': m, 'actors': []}
        az = np.radians(np.arange(-20, 20, 0.2))
        el = np.radians(np.arange(-15, -5, 0.5))
        A, E = np.meshgrid(az, el)
        d = np.c_[np.cos(E.ravel()) * np.cos(A.ravel()), np.cos(E.ravel()) * np.sin(A.ravel()), np.sin(E.ravel())]
        rng_r = 3.2 / -d[:, 2]  # flat ground at -3.2
        raw = np.c_[d * rng_r[:, None], np.ones(len(d))].astype(np.float32)
        raw[:, 1] *= -1  # store as CARLA left-handed
        cont = [[15.0, 0.0, -3.2 + 1.25, 8.0, 3.0, 2.5, 0.0]]
        p1, gt1, det = C.convert_frame(raw, rec, cont, [1, 2, 3])
        p2, gt2, _ = C.convert_frame(raw, rec, cont, [1, 2, 3])
        self.assertTrue(np.array_equal(p1, p2))
        self.assertEqual(gt1, gt2)
        self.assertEqual([g['class'] for g in gt1], ['truck'])
        self.assertGreater(det[0]['hits'], 20)


if __name__ == '__main__':
    unittest.main()
