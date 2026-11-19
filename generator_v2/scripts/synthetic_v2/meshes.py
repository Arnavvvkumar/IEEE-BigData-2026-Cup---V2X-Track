"""Procedural triangle meshes.

All builders return a Mesh in an object-local frame: x forward (heading), y left,
z up. Road-user builders return meshes whose axis-aligned bounds are centered in
x/y with the lowest vertex at z = 0, so the annotation box is exactly the mesh
bounds. Per-triangle reflectivity drives the return-strength dropout model.

Limitations (honest): shapes are lofted cross-sections, boxes, cylinders and
ellipsoids. There are no wheel wells, mirrors, open pickup beds, interiors,
transparent glass, articulated trailers or cloth. Pedestrians are rigid limb
assemblies in a single walking/standing pose.
"""
import math
from dataclasses import dataclass

import numpy as np

from .geometry import convex_hull, rot_y, rot_z


@dataclass
class Mesh:
    vertices: np.ndarray      # (N, 3) float64
    triangles: np.ndarray     # (M, 3) int32
    reflectivity: np.ndarray  # (M,) float32 in (0, 1]

    def transformed(self, rotation=None, translation=(0.0, 0.0, 0.0)):
        v = self.vertices if rotation is None else self.vertices @ np.asarray(rotation).T
        return Mesh(v + np.asarray(translation, np.float64), self.triangles, self.reflectivity)

    def with_reflectivity(self, value):
        return Mesh(self.vertices, self.triangles, np.full(len(self.triangles), value, np.float32))

    @property
    def bounds(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def merge(meshes):
    vertices, triangles, refl, offset = [], [], [], 0
    for m in meshes:
        vertices.append(m.vertices)
        triangles.append(m.triangles + offset)
        refl.append(m.reflectivity)
        offset += len(m.vertices)
    return Mesh(np.vstack(vertices), np.vstack(triangles).astype(np.int32), np.concatenate(refl).astype(np.float32))


def _mesh(v, t, refl):
    t = np.asarray(t, np.int32)
    return Mesh(np.asarray(v, np.float64), t, np.full(len(t), refl, np.float32))


def box(size, center=(0.0, 0.0, 0.0), refl=0.3):
    hx, hy, hz = (s / 2 for s in size)
    v = np.array([[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                  [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]]) + np.asarray(center)
    t = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return _mesh(v, t, refl)


def prism(polygon, z0, z1, refl=0.3):
    """Vertical extrusion of a convex 2-D polygon between z0 and z1."""
    poly = convex_hull(polygon)
    k = len(poly)
    v = np.vstack([np.column_stack([poly, np.full(k, z0)]), np.column_stack([poly, np.full(k, z1)])])
    t = []
    for i in range(1, k - 1):
        t += [[0, i + 1, i], [k, k + i, k + i + 1]]
    for i in range(k):
        j = (i + 1) % k
        t += [[i, j, k + j], [i, k + j, k + i]]
    return _mesh(v, t, refl)


def cylinder(radius, height, segments=16, base=(0.0, 0.0, 0.0), radius_top=None, refl=0.3, caps=True):
    """Vertical (optionally tapered) cylinder whose bottom center is `base`."""
    radius_top = radius if radius_top is None else radius_top
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.column_stack([np.cos(a), np.sin(a)])
    v = np.vstack([np.column_stack([ring * radius, np.zeros(segments)]),
                   np.column_stack([ring * radius_top, np.full(segments, height)]),
                   [[0, 0, 0], [0, 0, height]]]) + np.asarray(base)
    t = []
    for i in range(segments):
        j = (i + 1) % segments
        t += [[i, j, segments + j], [i, segments + j, segments + i]]
        if caps:
            t += [[2 * segments, j, i], [2 * segments + 1, segments + i, segments + j]]
    return _mesh(v, t, refl)


def ellipsoid(radii, center=(0.0, 0.0, 0.0), rings=8, segments=14, refl=0.3, rng=None, jitter=0.0):
    """UV ellipsoid; optional multiplicative radial jitter for irregular foliage."""
    v = [[0, 0, -1]]
    for i in range(1, rings):
        phi = -np.pi / 2 + np.pi * i / rings
        for j in range(segments):
            th = 2 * np.pi * j / segments
            v.append([math.cos(phi) * math.cos(th), math.cos(phi) * math.sin(th), math.sin(phi)])
    v.append([0, 0, 1])
    v = np.asarray(v, np.float64)
    if rng is not None and jitter > 0:
        v *= 1 + rng.uniform(-jitter, jitter, (len(v), 1))
    v = v * np.asarray(radii) + np.asarray(center)
    top = len(v) - 1
    t = []
    for j in range(segments):
        t.append([0, 1 + (j + 1) % segments, 1 + j])
    for i in range(rings - 2):
        r0, r1 = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            jn = (j + 1) % segments
            t += [[r0 + j, r0 + jn, r1 + jn], [r0 + j, r1 + jn, r1 + j]]
    last = 1 + (rings - 2) * segments
    for j in range(segments):
        t.append([top, last + j, last + (j + 1) % segments])
    return _mesh(v, t, refl)


def limb(start, end, r0, r1, segments=10, refl=0.3):
    """Tapered cylinder between two 3-D points."""
    start, end = np.asarray(start, np.float64), np.asarray(end, np.float64)
    axis = end - start
    length = float(np.linalg.norm(axis))
    m = cylinder(r0, length, segments, radius_top=r1, refl=refl)
    z = axis / length
    helper = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return m.transformed(np.column_stack([x, y, z]), start)


def loft(xs, sections, refl=0.3):
    """Loft along x through convex y-z sections (each (k, 2), same k), capped."""
    n, k = len(xs), len(sections[0])
    v = np.vstack([np.column_stack([np.full(k, x), s]) for x, s in zip(xs, sections)])
    t = []
    for i in range(n - 1):
        a, b = i * k, (i + 1) * k
        for j in range(k):
            jn = (j + 1) % k
            t += [[a + j, b + j, b + jn], [a + j, b + jn, a + jn]]
    last = (n - 1) * k
    for j in range(1, k - 1):
        t += [[0, j + 1, j], [last, last + j, last + j + 1]]
    return _mesh(v, t, refl)


def wheel(radius, width, center, refl=0.05):
    m = cylinder(radius, width, 16, base=(0, 0, -width / 2), refl=refl)
    # Rotate the z-axis cylinder so its axis points along y.
    rotation = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    return m.transformed(rotation, center)


def normalize_object(mesh):
    """Shift so x/y bounds are centered and the lowest point is z = 0."""
    lo, hi = mesh.bounds
    return mesh.transformed(None, (-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))


# ---------------------------------------------------------------- road users

CAR_SUBTYPES = {
    #            L range     W range      H range      belt frac  profile keypoints (u from rear, z as fraction of H; -1 => belt)
    "sedan":     ((4.40, 5.00), (1.75, 1.90), (1.38, 1.50), 0.58,
                  [(0.0, -0.80), (0.02, -1.0), (0.20, -1.04), (0.34, 1.0), (0.60, 1.0), (0.76, -1.03), (0.97, -0.92), (1.0, -0.75)]),
    "hatchback": ((3.80, 4.35), (1.68, 1.82), (1.42, 1.56), 0.58,
                  [(0.0, -0.85), (0.03, 0.90), (0.08, 1.0), (0.55, 1.0), (0.72, -1.03), (0.97, -0.92), (1.0, -0.75)]),
    "suv":       ((4.40, 5.10), (1.80, 2.00), (1.64, 1.86), 0.60,
                  [(0.0, -0.85), (0.02, 0.93), (0.06, 1.0), (0.62, 1.0), (0.75, -1.02), (0.97, -0.95), (1.0, -0.80)]),
    "pickup":    ((5.20, 5.90), (1.90, 2.05), (1.78, 1.95), 0.62,
                  [(0.0, -0.95), (0.02, -1.0), (0.36, -1.0), (0.38, 0.97), (0.40, 1.0), (0.60, 1.0), (0.70, -1.02), (0.97, -0.96), (1.0, -0.85)]),
    "minivan":   ((4.80, 5.20), (1.88, 2.00), (1.70, 1.80), 0.55,
                  [(0.0, -0.85), (0.02, 0.92), (0.05, 1.0), (0.68, 1.0), (0.82, -1.04), (0.97, -0.94), (1.0, -0.80)]),
}


def _car_section(w, zb, zbelt, zt, roof_frac):
    top = max(zt, zb + 0.2)
    side = min(zbelt, top)
    wr = w * roof_frac if top > zbelt + 1e-6 else w * 0.97
    return np.array([[-w * 0.92, zb], [-w, zb + 0.12], [-w, side], [-wr, top],
                     [wr, top], [w, side], [w, zb + 0.12], [w * 0.92, zb]])


def car(rng, subtype=None, paint=None):
    subtype = subtype or rng.choice(list(CAR_SUBTYPES))
    (l0, l1), (w0, w1), (h0, h1), belt_frac, keys = CAR_SUBTYPES[subtype]
    L, W, H = rng.uniform(l0, l1), rng.uniform(w0, w1), rng.uniform(h0, h1)
    clearance = rng.uniform(0.15, 0.24) if subtype in ("suv", "pickup") else rng.uniform(0.12, 0.18)
    zbelt = H * (belt_frac + rng.uniform(-0.03, 0.03))
    ku = np.array([k[0] for k in keys])
    kz = np.array([zbelt * -k[1] if k[1] < 0 else H * k[1] for k in keys])
    us = np.unique(np.concatenate([np.linspace(0, 1, 26), ku]))
    zs = np.interp(us, ku, kz)
    # Plan-view rounding at bumpers.
    taper = np.clip(np.minimum(us, 1 - us) / 0.06, 0, 1)
    widths = W / 2 * (0.88 + 0.12 * np.sqrt(taper))
    roof = rng.uniform(0.76, 0.86)
    sections = [_car_section(w, clearance, zbelt, z, roof) for w, z in zip(widths, zs)]
    body = loft(-L / 2 + us * L, sections, refl=1.0)
    paint = float(paint if paint is not None else rng.choice([rng.uniform(0.03, 0.12), rng.uniform(0.15, 0.85)]))
    centroids_z = body.vertices[body.triangles].mean(axis=1)[:, 2]
    body.reflectivity = np.where(centroids_z > zbelt + 0.03, 0.06, paint).astype(np.float32)
    r = rng.uniform(0.31, 0.34) if subtype in ("sedan", "hatchback") else rng.uniform(0.35, 0.41)
    overhang = rng.uniform(0.75, 1.0)
    wheels = [wheel(r, 0.22, (sx * (L / 2 - overhang - r * 0.3), sy * (W / 2 - 0.14), r))
              for sx in (-1, 1) for sy in (-1, 1)]
    return normalize_object(merge([body] + wheels)), subtype


TRUCK_SUBTYPES = ("box_truck", "bus", "semi", "cargo_van")


def truck(rng, subtype=None):
    subtype = subtype or rng.choice(TRUCK_SUBTYPES, p=[0.35, 0.25, 0.2, 0.2])
    paint = float(rng.uniform(0.2, 0.85))
    parts = []
    if subtype == "cargo_van":
        L, W, H = rng.uniform(5.5, 6.6), rng.uniform(2.0, 2.1), rng.uniform(2.4, 2.8)
        keys = [(0.0, 0.97 * H), (0.02, H), (0.82, H), (0.90, 0.55 * H), (0.98, 0.42 * H), (1.0, 0.36 * H)]
        us = np.unique(np.concatenate([np.linspace(0, 1, 20), [k[0] for k in keys]]))
        zs = np.interp(us, [k[0] for k in keys], [k[1] for k in keys])
        sections = [_car_section(W / 2, 0.2, 0.55 * H, z, 0.93) for z in zs]
        parts.append(loft(-L / 2 + us * L, sections, refl=paint))
        r, axles = 0.38, (-L / 2 + 1.0, L / 2 - 1.0)
    elif subtype == "bus":
        L, W, H = rng.uniform(10.0, 12.5), rng.uniform(2.50, 2.55), rng.uniform(3.0, 3.3)
        body = box((L, W, H - 0.32), (0, 0, 0.32 + (H - 0.32) / 2), refl=paint)
        cz = body.vertices[body.triangles].mean(axis=1)[:, 2]
        body.reflectivity = np.where((cz > 1.3) & (cz < H - 0.4), 0.08, paint).astype(np.float32)
        parts.append(body)
        r, axles = 0.5, (-L / 2 + 3.0, L / 2 - 2.6)
    elif subtype == "semi":
        cab_l, trailer_l = rng.uniform(5.6, 6.6), rng.uniform(10.5, 14.6)
        gap = rng.uniform(0.6, 1.1)
        L, W, H = cab_l + trailer_l - 1.5, 2.55, rng.uniform(3.8, 4.1)
        x_front = L / 2
        cab_h = rng.uniform(3.0, 3.6)
        parts.append(box((2.4, 2.45, cab_h - 0.9), (x_front - 1.2, 0, 0.9 + (cab_h - 0.9) / 2), refl=paint))
        parts.append(box((cab_l - 2.4, 1.0, 0.35), (x_front - 2.4 - (cab_l - 2.4) / 2, 0, 1.0)))
        trailer_front = x_front - 2.4 - gap
        parts.append(box((trailer_l, W, H - 1.25), (trailer_front - trailer_l / 2, 0, 1.25 + (H - 1.25) / 2),
                         refl=float(rng.uniform(0.3, 0.9))))
        r = 0.52
        axles = (x_front - 1.4, x_front - cab_l + 1.2, trailer_front - trailer_l + 1.4, trailer_front - trailer_l + 2.7)
    else:  # box_truck
        cab_l, cargo_l = rng.uniform(2.0, 2.5), rng.uniform(4.5, 7.5)
        gap = rng.uniform(0.15, 0.4)
        L, W, H = cab_l + gap + cargo_l, rng.uniform(2.35, 2.55), rng.uniform(3.0, 3.8)
        cab_h = rng.uniform(2.5, min(3.0, H - 0.2))
        x_front = L / 2
        cab = loft(np.array([x_front - cab_l, x_front - 0.6, x_front - 0.05, x_front]),
                   [_car_section(W / 2 * 0.97, 0.6, cab_h * 0.55, z, 0.93)
                    for z in (cab_h, cab_h, cab_h * 0.62, cab_h * 0.5)], refl=paint)
        parts.append(cab)
        parts.append(box((cargo_l, W, H - 1.05), (-L / 2 + cargo_l / 2, 0, 1.05 + (H - 1.05) / 2),
                         refl=float(rng.uniform(0.4, 0.9))))
        parts.append(box((L - 0.6, 0.9, 0.3), (0, 0, 0.75)))
        r, axles = 0.47, (x_front - 1.2, -L / 2 + 1.6)
    parts += [wheel(r, 0.3, (x, s * (W / 2 - 0.2), r)) for x in axles for s in (-1, 1)]
    return normalize_object(merge(parts)), subtype


def pedestrian(rng, child_prob=0.1):
    child = rng.random() < child_prob
    H = rng.uniform(1.0, 1.4) if child else rng.uniform(1.52, 1.95)
    width = H * rng.uniform(0.22, 0.26)
    walking = rng.random() < 0.6
    phase = rng.uniform(0, 2 * np.pi) if walking else 0.0
    stride = math.radians(rng.uniform(12, 22)) if walking else math.radians(rng.uniform(0, 4))
    cloth_top, cloth_bottom = float(rng.uniform(0.08, 0.7)), float(rng.uniform(0.08, 0.6))
    hip_z, shoulder_z = 0.52 * H, 0.81 * H
    leg_len = hip_z - 0.04
    parts = []
    for side, sign in ((-1, 1), (1, -1)):
        ang = sign * stride * math.sin(phase)
        foot = np.array([leg_len * math.sin(ang), side * width * 0.2, hip_z - leg_len * math.cos(ang)])
        parts.append(limb((0, side * width * 0.2, hip_z), foot, 0.075 * H / 1.75, 0.05 * H / 1.75, refl=cloth_bottom))
        parts.append(box((0.24 * H / 1.75, 0.1, 0.07), foot + [0.05, 0, -0.01], refl=0.1))
        arm_ang = -ang * 0.8
        hand = np.array([0.36 * H * math.sin(arm_ang), side * width * 0.5, shoulder_z - 0.36 * H * math.cos(arm_ang)])
        parts.append(limb((0, side * width * 0.48, shoulder_z), hand, 0.042 * H / 1.75, 0.035 * H / 1.75, refl=cloth_top))
    torso_h = shoulder_z - hip_z + 0.06
    parts.append(ellipsoid((0.12 * H / 1.75, width * 0.5, torso_h / 2), (0, 0, hip_z + torso_h / 2 - 0.02),
                           rings=6, segments=12, refl=cloth_top))
    parts.append(limb((0, 0, shoulder_z), (0, 0, shoulder_z + 0.06 * H), 0.05, 0.05, refl=0.3))
    parts.append(ellipsoid((0.1 * H / 1.75, 0.08 * H / 1.75, 0.115 * H / 1.75), (0.01, 0, 0.93 * H - 0.005),
                           rings=6, segments=10, refl=float(rng.uniform(0.1, 0.5))))
    accessory = "none"
    if not child and rng.random() < 0.2:
        accessory = "backpack"
        parts.append(box((0.16, 0.3, 0.42), (-0.12 * H / 1.75 - 0.08, 0, hip_z + torso_h * 0.6), refl=float(rng.uniform(0.1, 0.6))))
    mesh = merge(parts)
    mesh = mesh.transformed(None, (0, 0, -mesh.bounds[0][2]))
    return normalize_object(mesh), ("child" if child else "adult") + ("_walking" if walking else "_standing") + (f"_{accessory}" if accessory != "none" else "")


# ---------------------------------------------------------------- static furniture

def tree(rng, base):
    trunk_h = rng.uniform(1.8, 3.2)
    trunk = cylinder(rng.uniform(0.12, 0.25), trunk_h + 0.5, 8, base=base, refl=0.35)
    rx = rng.uniform(1.4, 3.2)
    crown = ellipsoid((rx, rx * rng.uniform(0.85, 1.15), rng.uniform(1.4, 3.0)),
                      np.asarray(base) + [0, 0, trunk_h + rx * 0.8], rings=7, segments=12,
                      refl=0.5, rng=rng, jitter=0.18)
    return trunk, crown


def light_pole(rng, base, arm_dir, height=None):
    height = height or rng.uniform(7.0, 9.5)
    pole = cylinder(0.11, height, 10, base=base, radius_top=0.08, refl=0.4)
    arm_len = rng.uniform(1.2, 2.5)
    tip = np.asarray(base) + [arm_dir[0] * arm_len, arm_dir[1] * arm_len, height - 0.1]
    arm = limb(np.asarray(base) + [0, 0, height - 0.3], tip, 0.05, 0.04, 8, refl=0.4)
    lamp = box((0.6, 0.3, 0.15), tip - [0, 0, 0.1], refl=0.5).transformed(None)
    return merge([pole, arm, lamp])


def signal_pole(rng, base, arm_dir, arm_len):
    height = rng.uniform(5.8, 7.0)
    pole = cylinder(0.16, height, 12, base=base, radius_top=0.13, refl=0.4)
    tip = np.asarray(base) + [arm_dir[0] * arm_len, arm_dir[1] * arm_len, height - 0.4]
    arm = limb(np.asarray(base) + [0, 0, height - 0.4], tip, 0.09, 0.06, 8, refl=0.4)
    heads = []
    for f in np.linspace(0.45, 0.95, int(rng.integers(1, 4))):
        p = np.asarray(base) + [arm_dir[0] * arm_len * f, arm_dir[1] * arm_len * f, height - 1.1]
        heads.append(box((0.35, 0.35, 1.05), p, refl=0.15))
    heads.append(box((0.3, 0.3, 0.9), np.asarray(base) + [0, 0, 3.0], refl=0.15))
    return merge([pole, arm] + heads)


def oriented_box(size, center_xy, yaw, z0, refl):
    m = box(size, (0, 0, z0 + size[2] / 2), refl)
    return m.transformed(rot_z(yaw), (center_xy[0], center_xy[1], 0.0))


def fence(start, end, height, refl=0.35, posts_every=2.5, solid=True):
    start, end = np.asarray(start, np.float64), np.asarray(end, np.float64)
    d = end - start
    length = float(np.linalg.norm(d))
    yaw = math.atan2(d[1], d[0])
    mid = (start + end) / 2
    if solid:
        return oriented_box((length, 0.2, height), mid, yaw, 0.0, refl)
    parts = [oriented_box((length, 0.06, 0.08), mid, yaw, height - 0.08, refl),
             oriented_box((length, 0.06, 0.08), mid, yaw, height * 0.45, refl)]
    for f in np.linspace(0, 1, max(2, int(length / posts_every) + 1)):
        p = start + d * f
        parts.append(cylinder(0.05, height, 6, base=(p[0], p[1], 0.0), refl=refl))
    return merge(parts)


def bench(base, yaw, refl=0.3):
    seat = box((1.8, 0.45, 0.06), (0, 0, 0.45), refl)
    back = box((1.8, 0.06, 0.4), (0, -0.2, 0.75), refl)
    legs = [box((0.06, 0.4, 0.45), (x, 0, 0.225), refl) for x in (-0.8, 0.8)]
    return merge([seat, back] + legs).transformed(rot_z(yaw), base)


def bush(rng, base):
    r = rng.uniform(0.5, 1.2)
    return ellipsoid((r, r * rng.uniform(0.8, 1.3), rng.uniform(0.4, 0.9)), np.asarray(base) + [0, 0, 0.35],
                     rings=5, segments=10, refl=0.45, rng=rng, jitter=0.2)


def building(rng, center_xy, yaw, size_xy, height):
    """Box building with optional stepped upper floor and roof parapet."""
    refl = float(rng.uniform(0.2, 0.7))
    parts = [oriented_box((size_xy[0], size_xy[1], height), center_xy, yaw, 0.0, refl)]
    if rng.random() < 0.3:
        parts.append(oriented_box((size_xy[0] * 0.6, size_xy[1] * 0.6, rng.uniform(2.5, 6.0)),
                                  center_xy, yaw, height, refl))
    if rng.random() < 0.4:
        # Ground-floor awning/canopy along the front facade.
        c, s = math.cos(yaw), math.sin(yaw)
        off = size_xy[1] / 2 + 0.6
        front = (center_xy[0] - s * off, center_xy[1] + c * off)
        parts.append(oriented_box((size_xy[0] * 0.8, 1.2, 0.15), front, yaw, 3.0, 0.3))
    return merge(parts)


def rot_y_mesh(mesh, angle):
    return mesh.transformed(rot_y(angle))
