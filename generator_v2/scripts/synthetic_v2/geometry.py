"""Frames, rotations and 7-DOF box utilities.

Conventions
-----------
* World frame W: the layout frame, flat ground at z = 0, right-handed, z up.
* Sensor mount: position p (world) and rotation R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
  Beam directions are defined in the physical sensor frame (x forward, y left,
  z up, elevation positive upward). Positive pitch tilts the forward axis down.
* Output frame S ("sensor frame"): origin at the sensor, rotated by the mount yaw
  only, so it stays gravity-aligned. Every cloud and annotation is written in S.
  Mount roll/pitch therefore change which surfaces the beams reach but never
  tilt the 7-DOF boxes. Ground appears at z = -sensor_height in S.
* Box: [x, y, z, l, w, h, yaw] with (x, y, z) the geometric center, full sizes,
  l along the object's heading (+x local), yaw in radians, counter-clockwise.
* Upstream YAML object: location = center, center = [0, 0, 0], extent = half
  sizes [l/2, w/2, h/2], angle = [roll, yaw, pitch] in degrees.
"""
import math

import numpy as np


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def mount_rotation(roll, pitch, yaw):
    """World-from-physical-sensor rotation (radians)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def wrap_angle(a):
    """Wrap to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def world_to_sensor_points(points, sensor_xyz, sensor_yaw):
    """Map world points (N, 3) into the gravity-aligned sensor frame S."""
    return (np.asarray(points, np.float64) - np.asarray(sensor_xyz, np.float64)) @ rot_z(sensor_yaw)


def sensor_to_world_points(points, sensor_xyz, sensor_yaw):
    return np.asarray(points, np.float64) @ rot_z(sensor_yaw).T + np.asarray(sensor_xyz, np.float64)


def world_box_to_sensor(box, sensor_xyz, sensor_yaw):
    box = np.asarray(box, np.float64)
    center = world_to_sensor_points(box[None, :3], sensor_xyz, sensor_yaw)[0]
    return np.concatenate([center, box[3:6], [float(wrap_angle(box[6] - sensor_yaw))]])


def box_to_yaml(box, obj_type, speed=0.0):
    """7-DOF sensor-frame box -> upstream V2X-Real object dictionary."""
    x, y, z, l, w, h, yaw = (float(v) for v in box)
    return {"location": [x, y, z], "center": [0.0, 0.0, 0.0],
            "extent": [l / 2, w / 2, h / 2], "angle": [0.0, math.degrees(yaw), 0.0],
            "obj_type": obj_type, "speed": float(speed)}


def yaml_to_box(obj):
    """Inverse of box_to_yaml (assumes zero roll/pitch, as written by this generator)."""
    loc = np.asarray(obj["location"], np.float64) + np.asarray(obj["center"], np.float64)
    ext = np.asarray(obj["extent"], np.float64) * 2
    return np.array([*loc, *ext, math.radians(obj["angle"][1])], np.float64)


def box_corners_bev(box, margin=0.0):
    """Four BEV corners (4, 2) of a box, counter-clockwise."""
    x, y, _, l, w, _, yaw = box[:7]
    hl, hw = l / 2 + margin, w / 2 + margin
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    c, s = math.cos(yaw), math.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def box_corners_3d(box):
    """Eight corners (8, 3): bottom four then top four."""
    bev = box_corners_bev(box)
    z0, z1 = box[2] - box[5] / 2, box[2] + box[5] / 2
    return np.vstack([np.column_stack([bev, np.full(4, z0)]), np.column_stack([bev, np.full(4, z1)])])


def polygons_overlap(a, b):
    """Separating-axis test for two convex polygons (k, 2)."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        for n in normals:
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def boxes_overlap_bev(a, b, margin=0.0):
    return polygons_overlap(box_corners_bev(a, margin), box_corners_bev(b, margin))


def points_to_box_local(points, box):
    """Express points in the box's local frame (origin at center, x along heading)."""
    return (np.asarray(points, np.float64) - np.asarray(box[:3], np.float64)) @ rot_z(box[6])


def points_in_box(points, box, tol=0.0):
    if len(points) == 0:
        return np.zeros(0, bool)
    local = points_to_box_local(points[:, :3], box)
    return (np.abs(local) <= np.asarray(box[3:6]) / 2 + tol).all(axis=1)


def ray_box_entry(directions, box, origin=(0.0, 0.0, 0.0)):
    """Slab-method entry distance of rays (origin + t*d, t >= 0) into an oriented box.

    Returns inf where the ray misses. Rays starting inside return 0.
    """
    rot = rot_z(box[6])
    o = (np.asarray(origin, np.float64) - np.asarray(box[:3], np.float64)) @ rot
    d = np.asarray(directions, np.float64) @ rot
    half = np.asarray(box[3:6], np.float64) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(np.abs(d) < 1e-12, 1e-12, d)
        t1 = (-half - o) / safe
        t2 = (half - o) / safe
    near = np.minimum(t1, t2).max(axis=1)
    far = np.maximum(t1, t2).min(axis=1)
    hit = far >= np.maximum(near, 0.0)
    return np.where(hit, np.maximum(near, 0.0), np.inf)


def convex_hull(points):
    """Monotone-chain convex hull of 2-D points, counter-clockwise, no repeats."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts, np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], np.float64)
