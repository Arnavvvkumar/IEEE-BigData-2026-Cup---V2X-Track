"""Ray-cast synthetic box-shaped containers (large dumpster / roll-off container, label 'truck') into CARLA scans.

Fully synthetic: the geometry is a parametric cuboid, inserted into synthetic CARLA returns. Only the *size range* is
informed by permitted V2X-Real box statistics (the 'TrashCan' subtype of the official truck superclass is ~8 x 2.9 x 2.5 m).
Each native return is a ray from the sensor origin; if that ray meets the cuboid before its recorded range, the return is
moved to the cuboid surface (+ the profile's 0.02 m range noise). A container is static for a whole clip (sensor static),
placed on free ground: no non-ground return inside it and no actor box within 1 m in any frame of the clip.
"""
import math

import numpy as np

DIMS = {'l': (6.0, 11.0), 'w': (2.4, 3.3), 'h': (2.2, 2.9)}
NOISE = 0.02


def ray_box_t(dirs, box):
    """Slab test. dirs: (N,3) ray vectors from the origin (ray = t * dirs, t in [0,1] reaches the native return).
    Returns entry parameter t (np.inf where the ray misses the box)."""
    x, y, z, l, w, h, yaw = box
    c, s = math.cos(-yaw), math.sin(-yaw)
    o = np.array([-x, -y, -z])
    ol = np.array([c * o[0] - s * o[1], s * o[0] + c * o[1], o[2]])
    dl = np.c_[c * dirs[:, 0] - s * dirs[:, 1], s * dirs[:, 0] + c * dirs[:, 1], dirs[:, 2]]
    half = np.array([l / 2, w / 2, h / 2])
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-half - ol) / dl
        t2 = (half - ol) / dl
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= tmin) & (tmax > 0)
    return np.where(hit, np.maximum(tmin, 0), np.inf)


def insert(points_rh, box, rng):
    """points_rh: (N,4) native right-handed returns (uncropped). Returns (new points, number of replaced returns)."""
    p = points_rh.copy()
    t = ray_box_t(p[:, :3].astype(np.float64), box)
    m = t < 1.0
    if m.any():
        r = np.linalg.norm(p[m, :3], axis=1)
        tn = t[m] * (1 + rng.normal(0, NOISE, m.sum()) / np.maximum(r * t[m], 1e-6))
        p[m, :3] = p[m, :3] * tn[:, None]
    return p, int(m.sum())


def inside(points, box, margin=0.0):
    d = points[:, :3] - np.asarray(box[:3])
    c, s = math.cos(-box[6]), math.sin(-box[6])
    lx, ly = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]
    return (np.abs(lx) <= box[3] / 2 + margin) & (np.abs(ly) <= box[4] / 2 + margin) & (np.abs(d[:, 2]) <= box[5] / 2 + margin)


def bev_overlap(a, b, margin):
    """Conservative circle test on BEV footprints."""
    ra = math.hypot(a[3], a[4]) / 2 + margin
    rb = math.hypot(b[3], b[4]) / 2
    return math.hypot(a[0] - b[0], a[1] - b[1]) < ra + rb


def place(frames_pts, frames_boxes, rng, k, tries=300):
    """frames_pts: list of (N,4) RH native clouds of one clip; frames_boxes: list of lists of actor boxes (7,).
    Returns up to k container boxes."""
    out = []
    p0 = frames_pts[0]
    for _ in range(k):
        for _ in range(tries):
            l, w, h = (float(rng.uniform(*DIMS[d])) for d in 'lwh')
            r, a, yaw = float(rng.uniform(8, 38)), float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(-math.pi, math.pi))
            x, y = r * math.cos(a), r * math.sin(a)
            near = p0[(p0[:, 0] - x) ** 2 + (p0[:, 1] - y) ** 2 < 9]
            if len(near) < 20:
                continue
            g = float(np.percentile(near[:, 2], 10))
            near_g = near[np.abs(near[:, 2] - g) < 0.3]
            if len(near_g) < 10 or np.ptp(near_g[:, 2]) > 0.5:  # need flat, observed ground
                continue
            box = np.array([x, y, g + h / 2 + 0.02, l, w, h, yaw])
            if any(bev_overlap(box, b, 1.0) for fb in frames_boxes for b in fb) or any(bev_overlap(box, b, 1.0) for b in out):
                continue
            # free space: no return above the ground slab inside the footprint (+0.3 m) up to the roof (+0.3 m), any frame
            probe = np.array([x, y, g + 0.25 + (h + 0.05) / 2, l + 0.6, w + 0.6, h + 0.05, yaw])
            if any(inside(fp, probe).sum() > 3 for fp in frames_pts):
                continue
            out.append(box)
            break
    return out
