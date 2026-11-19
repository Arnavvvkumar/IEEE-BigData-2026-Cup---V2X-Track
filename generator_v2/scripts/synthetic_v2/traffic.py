"""Lane-, sidewalk- and crosswalk-aware placement of annotated road users.

Vehicles and trucks: travel lanes (heading = lane direction), queues near the
stop line of inbound lanes, turning/through paths across the junction
(quadratic Bezier, heading = path tangent), curbside parking, open-lot slots.
Pedestrians: sidewalks (walking along the arm or standing), crosswalks just
outside the junction, corner waiting areas; optional small groups.
All placements are rejected if their BEV boxes overlap other road users or
static obstacles, or (for vehicles) a raised curb polygon.

Limitations: no traffic-signal phase logic, no lane changes, no cyclists or
motorcycles, no articulated trailer angles, static snapshot (no motion skew).
"""
import math
from dataclasses import dataclass

import numpy as np

from . import meshes as M
from .geometry import box_corners_bev, boxes_overlap_bev, polygons_overlap
from .layout import _choice

OBJ_TYPES = {"vehicle": "Car", "pedestrian": "Pedestrian", "truck": "Truck"}


@dataclass
class RoadUser:
    cls: str
    subtype: str
    mesh: M.Mesh          # object-local, x/y-centered bounds, z from 0
    dims: np.ndarray      # (l, w, h) = mesh bounds
    x: float = 0.0
    y: float = 0.0
    ground_z: float = 0.0
    yaw: float = 0.0
    placement: str = ""
    speed: float = 0.0

    @property
    def obj_type(self):
        return OBJ_TYPES[self.cls]

    @property
    def box_world(self):
        return np.array([self.x, self.y, self.ground_z + self.dims[2] / 2, *self.dims, self.yaw], np.float64)


def build_user(cls, rng, cfg):
    if cls == "vehicle":
        mesh, sub = M.car(rng)
    elif cls == "truck":
        mesh, sub = M.truck(rng)
    else:
        mesh, sub = M.pedestrian(rng, cfg["traffic"]["child_prob"])
    lo, hi = mesh.bounds
    return RoadUser(cls=cls, subtype=str(sub), mesh=mesh, dims=hi - lo)


def sample_counts(rng, cfg):
    tc = cfg["traffic"]
    if rng.random() < tc["empty_scene_prob"]:
        return {"vehicle": 0, "pedestrian": 0, "truck": 0}, True
    counts = {}
    for cls in ("vehicle", "pedestrian", "truck"):
        lo, hi = tc["count_range"][cls]
        counts[cls] = 0 if rng.random() < tc["class_absent_prob"][cls] else int(rng.integers(lo, hi + 1))
    return counts, False


def _bezier(p0, p1, p2, t):
    pos = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    tan = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1)
    return pos, math.atan2(tan[1], tan[0])


def _junction_pose(layout, rng, user):
    arms = layout.arms
    ia, ib = rng.choice(len(arms), 2, replace=False)
    a, b = arms[ia], arms[ib]
    ka, kb = int(rng.integers(a.lanes)), int(rng.integers(b.lanes))
    p0 = a.point(a.start, a.lane_lateral(False, ka))
    p2 = b.point(b.start, b.lane_lateral(True, kb))
    d0, d2 = -a.u, b.u
    mat = np.column_stack([d0, -d2])
    p1 = (p0 + p2) / 2
    if abs(np.linalg.det(mat)) > 0.05:
        t = np.linalg.solve(mat, p2 - p0)
        if t[0] > 0 and t[1] > 0:
            p1 = p0 + t[0] * d0
    pos, yaw = _bezier(p0, p1, p2, rng.uniform(0.1, 0.9))
    return pos, yaw, 0.0


def _vehicle_pose(layout, rng, kind, user, cfg):
    jit = math.radians(cfg["traffic"]["heading_jitter_deg"]["lane"])
    arm = layout.arms[int(rng.integers(len(layout.arms)))]
    if kind == "junction" and layout.has_junction:
        return _junction_pose(layout, rng, user)
    if kind == "parking":
        if layout.lot_slots and (not any(a.parking for a in layout.arms) or rng.random() < 0.4):
            x, y, yaw = layout.lot_slots[int(rng.integers(len(layout.lot_slots)))]
            yaw += np.pi * int(rng.integers(2)) + rng.normal(0, math.radians(cfg["traffic"]["heading_jitter_deg"]["parking"]))
            return np.array([x, y]), yaw, 0.0
        parked = [a for a in layout.arms if a.parking]
        if parked:
            arm = parked[int(rng.integers(len(parked)))]
            right = bool(rng.integers(2))
            along = rng.uniform(arm.start + 6, 70)
            lat = arm.parking_lateral(right) + rng.normal(0, 0.1)
            yaw = arm.heading(right) + rng.normal(0, math.radians(cfg["traffic"]["heading_jitter_deg"]["parking"]))
            return arm.point(along, lat), yaw, 0.0
        kind = "lane"
    outbound = bool(rng.integers(2))
    k = int(rng.integers(arm.lanes))
    if user.cls == "truck" and rng.random() < 0.6:
        k = arm.lanes - 1  # trucks prefer the curb lane
    if kind == "queue" and layout.has_junction:
        outbound = False
        along = arm.start + 4.5 + user.dims[0] / 2 + rng.exponential(8.0)
    else:
        lo = arm.start + 1.0 + user.dims[0] / 2 if layout.has_junction else -70
        along = rng.uniform(lo, 70)
    lat = arm.lane_lateral(outbound, k) + float(np.clip(rng.normal(0, 0.2), -0.5, 0.5))
    return arm.point(along, lat), arm.heading(outbound) + rng.normal(0, jit), 0.0


def _point_in_poly(poly, rng):
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    for _ in range(50):
        p = rng.uniform(lo, hi)
        if polygons_overlap(np.array([p, p + 1e-3, p + [0, 1e-3]]), poly):
            return p
    return poly.mean(axis=0)


def _pedestrian_pose(layout, rng, kind, user, cfg):
    jit = math.radians(cfg["traffic"]["heading_jitter_deg"]["pedestrian"])
    walking = "walking" in user.subtype
    arm = layout.arms[int(rng.integers(len(layout.arms)))]
    if kind == "corner" and layout.corners:
        p = _point_in_poly(layout.corners[int(rng.integers(len(layout.corners)))], rng)
        return p, rng.uniform(-np.pi, np.pi), layout.curb_height
    if kind == "crosswalk":
        along = (arm.start + rng.uniform(0.8, 3.2)) if layout.has_junction else rng.uniform(-30, 30)
        lat = rng.uniform(-arm.half_width + 0.4, arm.half_width - 0.4)
        yaw = arm.angle + np.pi / 2 * rng.choice([-1, 1]) + rng.normal(0, jit * 0.6)
        return arm.point(along, lat), yaw, 0.0
    side = rng.choice([-1, 1])
    lo = arm.start + 0.5 if layout.has_junction else -70
    along = rng.uniform(lo, 70)
    lat = side * (arm.half_width + rng.uniform(0.35, max(0.4, arm.sidewalk - 0.35)))
    yaw = (arm.angle + np.pi * rng.integers(2) + rng.normal(0, jit)) if walking else rng.uniform(-np.pi, np.pi)
    return arm.point(along, lat), yaw, layout.curb_height


def _fits(user, placed, layout, sensor_xy, radius, raised_check):
    box = user.box_world
    if np.hypot(box[0] - sensor_xy[0], box[1] - sensor_xy[1]) > radius:
        return False
    for other in placed:
        margin = 0.15 if user.cls == other.cls == "pedestrian" else 0.35
        if boxes_overlap_bev(box, other.box_world, margin / 2):
            return False
    for obs in layout.obstacles:
        if boxes_overlap_bev(box, obs, 0.05):
            return False
    top = user.ground_z + user.dims[2]
    for canopy, z_bottom in layout.canopies:
        if top > z_bottom - 0.1 and boxes_overlap_bev(box, canopy, 0.05):
            return False
    if raised_check:
        corners = box_corners_bev(box)
        if any(polygons_overlap(corners, poly) for poly in layout.raised):
            return False
    # Never place anything on top of the sensor mount itself.
    if np.hypot(box[0] - sensor_xy[0], box[1] - sensor_xy[1]) < 1.0 + max(box[3], box[4]) / 2:
        return False
    return True


def place_traffic(layout, rng, cfg):
    tc = cfg["traffic"]
    counts, empty = sample_counts(rng, cfg)
    sensor_xy = layout.sensor["xyz"][:2]
    radius = tc["placement_radius_m"]
    placed, failed = [], {"vehicle": 0, "pedestrian": 0, "truck": 0}
    # Large objects first so they claim lane space before cars and people.
    queue = ["truck"] * counts["truck"] + ["vehicle"] * counts["vehicle"] + ["pedestrian"] * counts["pedestrian"]
    pending_group = []
    while queue or pending_group:
        cls = pending_group.pop() if pending_group else queue.pop(0)
        group_anchor = None
        if isinstance(cls, tuple):
            cls, group_anchor = cls
        user = build_user(cls, rng, cfg)
        ok = False
        for _ in range(tc["max_attempts"]):
            if group_anchor is not None:
                ang = rng.uniform(-np.pi, np.pi)
                pos = np.array([group_anchor.x, group_anchor.y]) + rng.uniform(0.6, 1.4) * np.array([math.cos(ang), math.sin(ang)])
                yaw, z = group_anchor.yaw + rng.normal(0, 0.3), group_anchor.ground_z
                kind = group_anchor.placement
            elif cls == "pedestrian":
                kind = _choice(rng, tc["pedestrian_kinds"])
                pos, yaw, z = _pedestrian_pose(layout, rng, kind, user, cfg)
            else:
                kind = _choice(rng, tc["vehicle_kinds"] if cls == "vehicle" else tc["truck_kinds"])
                pos, yaw, z = _vehicle_pose(layout, rng, kind, user, cfg)
            user.x, user.y, user.yaw, user.ground_z = float(pos[0]), float(pos[1]), float((yaw + np.pi) % (2 * np.pi) - np.pi), float(z)
            user.placement = kind
            if group_anchor is not None and kind == "sidewalk":
                # Keep group members on the same raised surface: reject if they drift onto the road.
                if not any(polygons_overlap(box_corners_bev(user.box_world), poly) for poly in layout.raised):
                    continue
            if _fits(user, placed, layout, sensor_xy, radius, raised_check=cls != "pedestrian"):
                ok = True
                break
        if not ok:
            failed[cls] += 1
            continue
        if cls == "pedestrian":
            user.speed = float(rng.uniform(0.8, 1.8)) if "walking" in user.subtype else 0.0
            if group_anchor is None and queue and queue[0] == "pedestrian" and rng.random() < tc["pedestrian_group_prob"]:
                for _ in range(min(int(rng.integers(1, 3)), queue.count("pedestrian"))):
                    queue.remove("pedestrian")
                    pending_group.append(("pedestrian", user))
        else:
            user.speed = 0.0 if user.placement in ("parking", "queue") else float(rng.uniform(2.0, 15.0))
        placed.append(user)
    return placed, {"requested": counts, "failed": failed, "empty_scene": empty}
