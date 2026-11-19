"""Road-layout sampling and static background construction (world frame).

A layout is a junction (or mid-block road) made of straight arms radiating from
the origin. Each arm has lanes per direction (US right-hand traffic), an
optional median and curbside parking, and raised sidewalks on both sides.
Corners between adjacent arms get a raised convex sidewalk fill. Lots behind
the sidewalks hold buildings, fences, hedges, or an open parking lot.

Limitations: ground is a flat plane, arms are straight, there are no lane
markings (they are paint, invisible to geometry), no driveways, ramps,
overpasses or terrain, and buildings are extruded boxes.
"""
import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import meshes as M
from .geometry import box_corners_bev, convex_hull, polygons_overlap

# Background point-label codes (objects use their positive IDs).
LABEL_GROUND, LABEL_CURB, LABEL_BUILDING, LABEL_VEGETATION, LABEL_POLE, LABEL_MOUNT = -1, -2, -3, -4, -5, -6
LABEL_NAMES = {LABEL_GROUND: "ground", LABEL_CURB: "sidewalk_curb_median", LABEL_BUILDING: "building_wall_fence",
               LABEL_VEGETATION: "vegetation", LABEL_POLE: "pole_sign_furniture", LABEL_MOUNT: "sensor_mount"}


@dataclass
class Arm:
    angle: float
    lanes: int
    lane_width: float
    median: float
    median_curb: bool
    parking: bool
    parking_width: float
    sidewalk: float
    length: float
    start: float = 0.0

    @property
    def u(self):
        return np.array([math.cos(self.angle), math.sin(self.angle)])

    @property
    def n(self):
        return np.array([-math.sin(self.angle), math.cos(self.angle)])

    @property
    def half_width(self):
        return self.median / 2 + self.lanes * self.lane_width + (self.parking_width if self.parking else 0.0)

    def point(self, along, lateral):
        return along * self.u + lateral * self.n

    def lane_lateral(self, outbound, k):
        """Lane-center lateral offset; outbound lanes are on the right (-n) side."""
        off = self.median / 2 + (k + 0.5) * self.lane_width
        return -off if outbound else off

    def parking_lateral(self, right_side):
        off = self.median / 2 + self.lanes * self.lane_width + self.parking_width / 2
        return -off if right_side else off

    def heading(self, outbound):
        return self.angle if outbound else self.angle + math.pi


@dataclass
class Layout:
    layout_id: int
    family: str
    arms: list
    curb_height: float
    sensor: dict
    static: list = field(default_factory=list)        # (Mesh, label, porosity)
    obstacles: list = field(default_factory=list)     # BEV boxes road users must avoid
    raised: list = field(default_factory=list)        # convex polygons of raised curbs (vehicles avoid)
    canopies: list = field(default_factory=list)      # (BEV AABB box, z_bottom) of porous foliage volumes
    corners: list = field(default_factory=list)       # convex corner polygons (pedestrian waiting areas)
    lot_slots: list = field(default_factory=list)     # (x, y, yaw) open-lot parking slots
    params: dict = field(default_factory=dict)

    @property
    def has_junction(self):
        return self.family != "straight"

    def fingerprint(self):
        blob = json.dumps(self.params, sort_keys=True, default=float)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _choice(rng, weights):
    keys = list(weights)
    p = np.asarray([weights[k] for k in keys], np.float64)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _road_params(rng, cfg):
    lc = cfg["layout"]
    lanes = int(rng.choice([1, 2, 3], p=np.asarray(lc["lanes_per_direction_weights"]) / sum(lc["lanes_per_direction_weights"])))
    median = float(rng.uniform(*lc["median_width_m"])) if rng.random() < lc["median_prob"] else 0.0
    return dict(lanes=lanes, lane_width=float(rng.uniform(*lc["lane_width_m"])), median=median,
                median_curb=bool(median > 0 and rng.random() < lc["median_curb_prob"]),
                parking=bool(lanes <= 2 and rng.random() < lc["parking_prob"]),
                parking_width=float(rng.uniform(*lc["parking_width_m"])),
                sidewalk=float(rng.uniform(*lc["sidewalk_width_m"])), length=float(lc["arm_length_m"]))


def _corner_pairs(arms):
    order = sorted(range(len(arms)), key=lambda i: arms[i].angle % (2 * np.pi))
    pairs = []
    for idx, i in enumerate(order):
        j = order[(idx + 1) % len(order)]
        gap = (arms[j].angle - arms[i].angle) % (2 * np.pi)
        pairs.append((i, j, gap))
    return pairs


def corner_polygon(a, b, gap):
    """Convex raised fill between arm a and the next arm b counter-clockwise."""
    p1 = a.point(a.start, a.half_width)
    p2 = a.point(a.start, a.half_width + a.sidewalk)
    p3 = b.point(b.start, -(b.half_width + b.sidewalk))
    p4 = b.point(b.start, -b.half_width)
    pts = [p1, p2, p3, p4]
    if gap < math.radians(179):
        # Intersection of the two outer sidewalk (property) lines.
        o1, d1 = a.point(0, a.half_width + a.sidewalk), a.u
        o2, d2 = b.point(0, -(b.half_width + b.sidewalk)), b.u
        mat = np.column_stack([d1, -d2])
        if abs(np.linalg.det(mat)) > 1e-6:
            t = np.linalg.solve(mat, o2 - o1)
            pts.append(o1 + t[0] * d1)
    return convex_hull(pts)


def sample_layout(layout_id, rng, cfg):
    lc = cfg["layout"]
    family = _choice(rng, lc["family_weights"])
    theta = float(rng.uniform(-np.pi, np.pi))
    jit = lambda: math.radians(rng.normal(0, lc["angle_jitter_deg"]))
    road_a, road_b = _road_params(rng, cfg), _road_params(rng, cfg)
    if family == "cross":
        cross = theta + np.pi / 2 + jit()
        spec = [(theta, road_a), (theta + np.pi, road_a), (cross, road_b), (cross + np.pi, road_b)]
    elif family == "skewed_cross":
        skew = math.radians(rng.uniform(*lc["skew_range_deg"])) * rng.choice([-1, 1])
        cross = theta + np.pi / 2 + skew
        spec = [(theta, road_a), (theta + np.pi, road_a), (cross, road_b), (cross + np.pi, road_b)]
    elif family == "tee":
        side = theta + np.pi / 2 + math.radians(rng.uniform(-1, 1) * lc["side_arm_jitter_deg"])
        spec = [(theta, road_a), (theta + np.pi, road_a), (side, road_b)]
    elif family == "y":
        road_c = _road_params(rng, cfg)
        spec = [(theta, road_a),
                (theta + math.radians(120 + rng.uniform(-1, 1) * lc["y_jitter_deg"]), road_b),
                (theta + math.radians(240 + rng.uniform(-1, 1) * lc["y_jitter_deg"]), road_c)]
    else:  # straight mid-block road
        spec = [(theta, road_a), (theta + np.pi, road_a)]
    arms = [Arm(angle=float((a + np.pi) % (2 * np.pi) - np.pi), **p) for a, p in spec]
    if family != "straight":
        hw = max(a.half_width + a.sidewalk for a in arms)
        need = [0.0] * len(arms)
        for i, j, gap in _corner_pairs(arms):
            if gap < math.radians(179):
                s = hw / math.tan(gap / 2) + 1.5
                need[i], need[j] = max(need[i], s), max(need[j], s)
        for arm, s in zip(arms, need):
            arm.start = float(max(s, hw * 0.5))
    curb = float(rng.uniform(*lc["curb_height_m"]))
    layout = Layout(layout_id=layout_id, family=family, arms=arms, curb_height=curb, sensor={})
    layout.sensor = sample_sensor(layout, rng, cfg)
    layout.params = {"family": family, "curb": round(curb, 4),
                     "arms": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in vars(a).items()} for a in arms],
                     "sensor": {k: (np.round(v, 4).tolist() if isinstance(v, (list, np.ndarray)) else v)
                                for k, v in layout.sensor.items() if k != "elevations_deg"}}
    return layout


def sample_sensor(layout, rng, cfg):
    sc = cfg["sensor"]
    placement = _choice(rng, sc["placement_weights"])
    if not layout.has_junction and placement == "corner":
        placement = "midblock"
    arm = layout.arms[int(rng.integers(len(layout.arms)))]
    side = float(rng.choice([-1, 1]))
    if placement == "corner":
        i, j, gap = _corner_pairs(layout.arms)[int(rng.integers(len(layout.arms)))]
        a, b = layout.arms[i], layout.arms[j]
        chord = (a.point(a.start, a.half_width) + b.point(b.start, -b.half_width)) / 2
        out = chord / max(np.linalg.norm(chord), 1e-6)
        xy = chord + out * rng.uniform(0.6, 1.8)
    elif placement == "mast":
        # Hanging from a mast arm above the road near the junction edge.
        lane_span = arm.half_width * 0.8
        xy = arm.point(arm.start + rng.uniform(-1.0, 4.0), rng.uniform(-lane_span, lane_span))
    else:
        xy = arm.point(arm.start + rng.uniform(2.0, 25.0), side * (arm.half_width + rng.uniform(0.4, max(0.5, arm.sidewalk - 0.3))))
    h = float(rng.uniform(*sc["height_m"]))
    lim = math.radians(sc["roll_pitch_max_deg"])
    roll, pitch = (float(np.clip(math.radians(rng.normal(0, sc["roll_pitch_sigma_deg"])), -lim, lim)) for _ in range(2))
    presets = sc["beam_presets"]
    p = np.asarray([b["weight"] for b in presets], np.float64)
    preset = presets[int(rng.choice(len(presets), p=p / p.sum()))]
    return {"placement": placement, "xyz": [float(xy[0]), float(xy[1]), h], "roll": roll, "pitch": pitch,
            "yaw": float(rng.uniform(-np.pi, np.pi)), "beam_preset": preset["name"],
            "mount_dir": float(rng.uniform(-np.pi, np.pi))}


def _corridors(layout, pad=0.0):
    """BEV boxes covering each arm's road + sidewalks (+pad), for lot exclusion."""
    out = []
    for a in layout.arms:
        width = 2 * (a.half_width + a.sidewalk + pad)
        center = a.point(a.length / 2, 0)
        out.append(np.array([center[0], center[1], 0, a.length + 2 * pad, width, 1, a.angle]))
    return out


def _add(layout, mesh, label, porosity=0.0):
    layout.static.append((mesh, label, porosity))
    if porosity > 0:
        # Porous foliage (crowns, bushes) is not a footprint obstacle, but road users must not
        # interpenetrate it: a pass-through re-cast could otherwise start inside an object.
        lo, hi = mesh.bounds
        c, size = (lo + hi) / 2, hi - lo
        layout.canopies.append((np.array([c[0], c[1], 0.0, size[0], size[1], 1.0, 0.0]), float(lo[2])))


def _obstacle(layout, xy, size, yaw=0.0):
    layout.obstacles.append(np.array([xy[0], xy[1], 0.0, size[0], size[1], 1.0, yaw]))


def build_static(layout, rng, cfg):
    bc = cfg["background"]
    curb = layout.curb_height
    ground = M.Mesh(np.array([[-250, -250, 0], [250, -250, 0], [250, 250, 0], [-250, 250, 0]], np.float64),
                    np.array([[0, 1, 2], [0, 2, 3]], np.int32), np.full(2, 0.12, np.float32))
    _add(layout, ground, LABEL_GROUND)

    for a in layout.arms:
        for side in (-1, 1):
            length = a.length - a.start
            c = a.point(a.start + length / 2, side * (a.half_width + a.sidewalk / 2))
            slab = M.oriented_box((length, a.sidewalk, curb), c, a.angle, 0.0, 0.3)
            _add(layout, slab, LABEL_CURB)
            layout.raised.append(box_corners_bev(np.array([c[0], c[1], 0, length, a.sidewalk, 1, a.angle])))
        if a.median_curb:
            length = a.length - a.start - 1.5
            c = a.point(a.start + 1.5 + length / 2, 0)
            _add(layout, M.oriented_box((length, a.median - 0.2, curb), c, a.angle, 0.0, 0.3), LABEL_CURB)
            layout.raised.append(box_corners_bev(np.array([c[0], c[1], 0, length, a.median - 0.2, 1, a.angle])))
            if a.median >= 2.5:
                for s in np.arange(a.start + 6, 70, rng.uniform(9, 16)):
                    xy = a.point(s, 0)
                    if rng.random() < 0.5:
                        trunk, crown = M.tree(rng, (xy[0], xy[1], curb))
                        _add(layout, trunk, LABEL_VEGETATION)
                        _add(layout, crown, LABEL_VEGETATION, bc["tree_crown_porosity"])
                    else:
                        _add(layout, M.bush(rng, (xy[0], xy[1], curb)), LABEL_VEGETATION, bc["bush_porosity"])

    if layout.has_junction:
        for i, j, gap in _corner_pairs(layout.arms):
            poly = corner_polygon(layout.arms[i], layout.arms[j], gap)
            _add(layout, M.prism(poly, 0.0, curb, 0.3), LABEL_CURB)
            layout.raised.append(poly)
            layout.corners.append(poly)
            if rng.random() < bc["signal_pole_prob"]:
                a = layout.arms[i]
                base_xy = a.point(a.start + 0.6, a.half_width + 0.6)
                mast = -a.n  # across a's inbound lanes
                _add(layout, M.signal_pole(rng, (base_xy[0], base_xy[1], curb), mast,
                                           a.half_width * rng.uniform(0.5, 0.9)), LABEL_POLE)
                _obstacle(layout, base_xy, (0.5, 0.5))

    _street_furniture(layout, rng, cfg)
    _lots(layout, rng, cfg)
    _sensor_mount(layout, cfg)
    return layout


def _street_furniture(layout, rng, cfg):
    bc = cfg["background"]
    curb = layout.curb_height
    for a in layout.arms:
        for side in (-1, 1):
            lat_curb = side * (a.half_width + 0.55)
            s = a.start + rng.uniform(3, 15)
            while s < 75:
                xy = a.point(s, lat_curb)
                _add(layout, M.light_pole(rng, (xy[0], xy[1], curb), -side * a.n), LABEL_POLE)
                _obstacle(layout, xy, (0.4, 0.4))
                s += rng.uniform(*bc["light_pole_spacing_m"])
            if rng.random() < bc["tree_side_prob"]:
                lat = side * (a.half_width + min(a.sidewalk * 0.55, 1.6))
                s = a.start + rng.uniform(4, 10)
                while s < 75:
                    xy = a.point(s, lat)
                    if all(np.hypot(*(xy - o[:2])) > 1.5 for o in layout.obstacles):
                        trunk, crown = M.tree(rng, (xy[0], xy[1], curb))
                        _add(layout, trunk, LABEL_VEGETATION)
                        _add(layout, crown, LABEL_VEGETATION, bc["tree_crown_porosity"])
                        _obstacle(layout, xy, (0.8, 0.8))
                    s += rng.uniform(*bc["tree_spacing_m"])
            n_items = rng.poisson(bc["furniture_per_100m"] * (75 - a.start) / 100)
            for _ in range(n_items):
                s = rng.uniform(a.start + 2, 75)
                kind = rng.choice(["bench", "hydrant", "bollard", "sign"] + (["bin"] if bc["trash_bins"] else []))
                outer = a.half_width + a.sidewalk - 0.5
                if kind == "bench":
                    xy = a.point(s, side * outer)
                    _add(layout, M.bench((xy[0], xy[1], curb), a.angle + (np.pi if side < 0 else 0)), LABEL_POLE)
                    _obstacle(layout, xy, (2.0, 0.7), a.angle)
                elif kind == "hydrant":
                    xy = a.point(s, side * (a.half_width + 0.45))
                    _add(layout, M.cylinder(0.14, 0.75, 10, (xy[0], xy[1], curb), refl=0.5), LABEL_POLE)
                    _obstacle(layout, xy, (0.4, 0.4))
                elif kind == "bollard":
                    for k in range(int(rng.integers(2, 6))):
                        xy = a.point(s + 1.4 * k, side * (a.half_width + 0.35))
                        _add(layout, M.cylinder(0.1, 0.9, 8, (xy[0], xy[1], curb), refl=0.4), LABEL_POLE)
                        _obstacle(layout, xy, (0.3, 0.3))
                elif kind == "sign":
                    xy = a.point(s, side * (a.half_width + 0.4))
                    post = M.cylinder(0.04, 2.6, 6, (xy[0], xy[1], curb), refl=0.4)
                    plate = M.oriented_box((0.05, 0.7, 0.7), xy, a.angle, curb + 2.0, 0.8)
                    _add(layout, M.merge([post, plate]), LABEL_POLE)
                    _obstacle(layout, xy, (0.3, 0.3))
                else:
                    xy = a.point(s, side * outer)
                    _add(layout, M.cylinder(0.3, 1.0, 10, (xy[0], xy[1], curb), refl=0.2), LABEL_POLE)
                    _obstacle(layout, xy, (0.7, 0.7))


def _lots(layout, rng, cfg):
    bc = cfg["background"]
    arms = layout.arms
    pairs = _corner_pairs(arms) if layout.has_junction else [(0, 1, np.pi), (1, 0, np.pi)]
    placed = []
    for i, j, gap in pairs:
        a, b = arms[i], arms[j]
        lot_kind = "parking" if rng.random() < bc["parking_lot_sector_prob"] else (
            "buildings" if rng.random() < bc["building_sector_prob"] else "open")
        setback = rng.uniform(*bc["building_setback_m"])
        corridors = _corridors(layout, pad=setback)
        if lot_kind == "parking":
            # Rows of perpendicular slots behind arm a's left (+n) sidewalk.
            base = a.half_width + a.sidewalk + rng.uniform(3, 6)
            road = _corridors(layout, pad=0.5)
            for row in range(int(rng.integers(1, 4))):
                lat = base + 2.6 + row * 6.0
                for s in np.arange(a.start + 6 + (0 if layout.has_junction else -40), a.start + 46, 2.7):
                    xy = a.point(s, lat)
                    slot = np.array([xy[0], xy[1], 0, 2.6, 5.0, 1, a.angle])
                    if any(polygons_overlap(box_corners_bev(slot), box_corners_bev(c)) for c in road) \
                            or np.linalg.norm(xy) < a.start:
                        continue
                    layout.lot_slots.append((float(xy[0]), float(xy[1]), float(a.angle + np.pi / 2)))
            if rng.random() < 0.5:
                start = a.point(a.start + 4, base - 1.0)
                end = a.point(a.start + 46, base - 1.0)
                _add(layout, M.fence(start, end, rng.uniform(0.8, 1.2), solid=False), LABEL_BUILDING)
            continue
        if lot_kind == "buildings":
            for _ in range(int(rng.integers(bc["buildings_per_sector"][0], bc["buildings_per_sector"][1] + 1))):
                for _attempt in range(25):
                    ang = a.angle + gap * rng.uniform(0.12, 0.88)
                    r = rng.uniform(a.start + 6, 75)
                    xy = np.array([math.cos(ang), math.sin(ang)]) * r
                    size = rng.uniform(*bc["building_footprint_m"], size=2)
                    yaw = rng.choice([a.angle, b.angle]) + rng.choice([0, np.pi / 2])
                    fp = np.array([xy[0], xy[1], 0, size[0], size[1], 1, yaw])
                    if any(polygons_overlap(box_corners_bev(fp), box_corners_bev(c)) for c in corridors):
                        continue
                    if any(polygons_overlap(box_corners_bev(fp, 1.0), box_corners_bev(p)) for p in placed):
                        continue
                    if np.linalg.norm(xy) < max(x.start for x in arms) + max(size) / 2:
                        continue
                    _add(layout, M.building(rng, xy, yaw, size, rng.uniform(*bc["building_height_m"])), LABEL_BUILDING)
                    placed.append(fp)
                    layout.obstacles.append(fp)
                    break
        # Property-line features behind arm a's left sidewalk.
        lat = a.half_width + a.sidewalk + 0.3
        s0, s1 = a.start + 2 + (0 if layout.has_junction else -60), a.start + rng.uniform(20, 60)
        if rng.random() < bc["fence_prob"]:
            _add(layout, M.fence(a.point(s0, lat), a.point(s1, lat), rng.uniform(1.0, 2.2),
                                 solid=bool(rng.random() < 0.5)), LABEL_BUILDING)
        elif rng.random() < bc["bush_prob"]:
            for s in np.arange(s0, s1, 1.6):
                xy = a.point(s, lat + 0.3)
                _add(layout, M.bush(rng, (xy[0], xy[1], 0.0)), LABEL_VEGETATION, bc["bush_porosity"])


def _sensor_mount(layout, cfg):
    if not cfg["sensor"]["mount_pole"]:
        return
    s = layout.sensor
    x, y, h = s["xyz"]
    d = np.array([math.cos(s["mount_dir"]), math.sin(s["mount_dir"])])
    if s["placement"] == "mast":
        # Horizontal mast arm passing 0.35 m above the sensor, plus a short hanger.
        p0 = np.array([x, y]) - d * 6.0
        p1 = np.array([x, y]) + d * 1.0
        mast = M.limb((p0[0], p0[1], h + 0.35), (p1[0], p1[1], h + 0.35), 0.1, 0.08, 8, refl=0.4)
        hanger = M.limb((x, y, h + 0.35), (x, y, h + 0.12), 0.03, 0.03, 6, refl=0.4)
        _add(layout, M.merge([mast, hanger]), LABEL_MOUNT)
    else:
        base = np.array([x, y]) + d * 0.6
        z0 = layout.curb_height if s["placement"] != "mast" else 0.0
        pole = M.cylinder(0.14, h + 0.6 - z0, 12, (base[0], base[1], z0), radius_top=0.11, refl=0.4)
        arm = M.limb((base[0], base[1], h + 0.1), (x, y, h + 0.1), 0.05, 0.05, 6, refl=0.4)
        _add(layout, M.merge([pole, arm]), LABEL_MOUNT)
        _obstacle(layout, base, (0.5, 0.5))
