"""Ground-plane fit from points alone, and the rigid transform that levels a scan. No labels involved."""
import numpy as np


def _lsq(P):
    return np.linalg.lstsq(np.c_[P[:, :2], np.ones(len(P))], P[:, 2], rcond=None)[0]


def fit_ground(pts, rng, iters=600, thr=0.12, c_range=(-7.0, -1.2), max_slope=0.35):
    """Plane z = a x + b y + c. Constrained RANSAC: the plane must pass well below the sensor (c in c_range) and not be steep.
    Candidates are drawn from the lowest points of a coarse XY grid so walls and beam cones do not dominate."""
    p = pts[(np.abs(pts[:, 0]) < 45) & (np.abs(pts[:, 1]) < 45) & np.isfinite(pts).all(1)]
    cell = (np.floor(p[:, 0] / 1.5).astype(np.int64) + 100) * 1000 + np.floor(p[:, 1] / 1.5).astype(np.int64) + 100
    order = np.lexsort((p[:, 2], cell))
    first = np.r_[True, cell[order][1:] != cell[order][:-1]]
    low = p[order][first]  # lowest point per 1.5 m cell
    sub = p[rng.choice(len(p), min(len(p), 30000), replace=False)]
    best, bn = None, -1
    for _ in range(iters):
        s = low[rng.choice(len(low), 3, replace=False)]
        try:
            co = _lsq(s)
        except np.linalg.LinAlgError:
            continue
        if not (c_range[0] <= co[2] <= c_range[1]) or np.hypot(co[0], co[1]) > max_slope:
            continue
        n = int((np.abs(low[:, 0] * co[0] + low[:, 1] * co[1] + co[2] - low[:, 2]) < thr).sum())
        if n > bn:
            bn, best = n, co
    if best is None:
        raise RuntimeError('no ground plane found')
    for _ in range(3):  # refine on all points near the plane
        inl = np.abs(sub[:, 0] * best[0] + sub[:, 1] * best[1] + best[2] - sub[:, 2]) < thr
        best = _lsq(sub[inl])
    return best


def level_transform(abc, target_ground_z):
    """R, t such that p' = R p + t maps the plane z = a x + b y + c to the level plane z = target_ground_z (minimal rotation)."""
    a, b, c = abc
    n = np.array([-a, -b, 1.0])
    norm = np.linalg.norm(n)
    n /= norm
    v = np.cross(n, [0, 0, 1.0])
    s, cth = np.linalg.norm(v), n[2]
    if s < 1e-12:
        R = np.eye(3)
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - cth) / s ** 2)
    h = c / norm  # plane height after rotation
    return R, np.array([0.0, 0.0, target_ground_z - h])
