"""Beam patterns, Open3D (Embree, CPU) ray casting and the measurement model.

Measurement order per ray:
  1. beam direction = channel elevation (+calibration jitter) x azimuth (+offset),
     rotated by the mount roll/pitch/yaw, plus small per-ray angular jitter;
  2. first-surface hit via RaycastingScene.cast_rays; porous surfaces (foliage)
     let a ray continue with probability `porosity` (re-cast just behind the hit);
  3. min/max range gating;
  4. return-strength detection: s = refl*|cos(incidence)|/(r/ref)^2,
     p = sigmoid(ln(s/threshold)/softness);
  5. i.i.d. random dropout and dead channels;
  6. Gaussian range noise along the ray (sigma0 + sigma_per_m * r);
  7. points expressed in the level sensor frame and cropped to the output range.

Limitations: single return, no beam divergence/mixed pixels, no rolling-shutter
motion skew, no multipath, no rain/fog, glass is opaque but dark.
"""
import math

import numpy as np
import open3d as o3d

from .geometry import mount_rotation, rot_z


def elevations(preset, rng, jitter_deg):
    if preset["pattern"] == "explicit":
        e = np.asarray(preset["elevations_deg"], np.float64)
    else:
        lo, hi = preset["elevation_deg"]
        n = int(preset["channels"])
        if preset["pattern"] == "uniform":
            e = np.linspace(lo, hi, n)
        else:  # gradient: denser near center_deg
            c, p = float(preset["center_deg"]), float(preset["power"])
            u = np.linspace(-1, 1, n)
            e = np.where(u < 0, c - (c - lo) * np.abs(u) ** p, c + (hi - c) * np.abs(u) ** p)
    return np.radians(np.sort(e) + rng.normal(0, jitter_deg, len(e)))


def beam_directions(elev, azimuth_steps, az_offsets):
    """Physical-frame unit directions (C*A, 3) and channel index per ray."""
    az = np.linspace(-np.pi, np.pi, int(azimuth_steps), endpoint=False)
    a = az[None, :] + az_offsets[:, None]
    e = np.broadcast_to(elev[:, None], a.shape)
    d = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1).reshape(-1, 3)
    channel = np.repeat(np.arange(len(elev)), len(az))
    return d, channel


def jitter_directions(d, rng, sigma_rad):
    if sigma_rad <= 0:
        return d
    d = d + rng.normal(0, sigma_rad, d.shape)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


class RayScene:
    """Thin wrapper over Open3D RaycastingScene with per-geometry metadata."""

    def __init__(self, nthreads=2):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.nthreads = int(nthreads)
        self.labels, self.porosity, self.refl = [], [], []

    def add(self, mesh, label, porosity=0.0):
        gid = self.scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)),
                                       o3d.core.Tensor(mesh.triangles.astype(np.uint32)))
        if gid != len(self.labels):
            raise RuntimeError("Unexpected Open3D geometry id ordering")
        self.labels.append(int(label))
        self.porosity.append(float(porosity))
        self.refl.append(np.asarray(mesh.reflectivity, np.float32))
        return gid

    def _cast(self, origins, dirs):
        rays = np.hstack([origins, dirs]).astype(np.float32)
        r = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=self.nthreads)
        t = r["t_hit"].numpy().astype(np.float64)
        gid = r["geometry_ids"].numpy().astype(np.int64)
        prim = r["primitive_ids"].numpy().astype(np.int64)
        normal = r["primitive_normals"].numpy().astype(np.float64)
        miss = ~np.isfinite(t) | (gid == o3d.t.geometry.RaycastingScene.INVALID_ID)
        t[miss] = np.inf
        gid[miss] = -1
        return t, gid, prim, normal

    def cast(self, origin, dirs, rng=None, max_passes=1):
        """First returns along dirs from a single origin, honoring porosity."""
        n = len(dirs)
        origins = np.broadcast_to(np.asarray(origin, np.float64), (n, 3)).copy()
        t_total = np.zeros(n)
        t, gid, prim, normal = self._cast(origins, dirs)
        porosity = np.asarray(self.porosity + [0.0])  # index -1 -> miss -> 0
        for _ in range(max_passes - 1):
            if rng is None:
                break
            through = (gid >= 0) & (rng.random(n) < porosity[gid])
            if not through.any():
                break
            step = t[through] + 0.02
            t_total[through] += step
            o2 = origins[through] + dirs[through] * step[:, None]
            t2, g2, p2, n2 = self._cast(o2, dirs[through])
            origins[through] = o2
            t[through], gid[through], prim[through], normal[through] = t2, g2, p2, n2
        t = np.where(gid >= 0, t_total + t, np.inf)
        refl = np.zeros(n)
        hit = gid >= 0
        offsets = np.concatenate([[0], np.cumsum([len(r) for r in self.refl])])
        refl[hit] = np.concatenate(self.refl)[offsets[gid[hit]] + prim[hit]]
        nn = np.linalg.norm(normal, axis=1)
        cos_inc = np.where(hit & (nn > 0), np.abs((normal * dirs).sum(axis=1)) / np.maximum(nn, 1e-12), 0.0)
        label = np.where(hit, np.asarray(self.labels + [0])[gid], 0)
        return t, label, refl, cos_inc


def simulate(scene, layout, cfg, rng, preset_rng):
    """Cast the layout's sensor and apply the measurement model.

    Returns dict with xyzi (float32, sensor frame), point labels, and the
    per-frame measurement parameters actually used.
    """
    sc, nc, dc = cfg["sensor"], cfg["noise"], cfg["dropout"]
    s = layout.sensor
    preset = next(p for p in sc["beam_presets"] if p["name"] == s["beam_preset"])
    elev = elevations(preset, preset_rng, sc["elevation_jitter_deg"])
    az_off = np.radians(preset_rng.normal(0, sc["azimuth_offset_jitter_deg"], len(elev)))
    d_phys, channel = beam_directions(elev, preset.get("azimuth_steps", 1024), az_off)
    d_phys = jitter_directions(d_phys, rng, math.radians(nc["angular_sigma_deg"]))
    rot = mount_rotation(s["roll"], s["pitch"], s["yaw"])
    d_world = d_phys @ rot.T
    t, label, refl, cos_inc = scene.cast(s["xyz"], d_world, rng, dc["max_passes"])

    sigma0 = float(rng.uniform(*nc["range_sigma_m"]))
    p_drop = float(rng.uniform(*dc["random_prob"]))
    dead = rng.random(len(elev)) < dc["dead_channel_prob"]
    keep = np.isfinite(t) & (t >= sc["min_range_m"]) & (t <= sc["max_range_m"])
    rm = dc["return_model"]
    if rm["enabled"]:
        with np.errstate(divide="ignore", invalid="ignore"):
            strength = refl * cos_inc / np.maximum(t / rm["ref_range_m"], 1e-6) ** 2
            logit = np.log(np.maximum(strength, 1e-12) / rm["threshold"]) / rm["softness"]
        p_detect = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
        keep &= rng.random(len(t)) < p_detect
    else:
        strength = refl * cos_inc
    keep &= rng.random(len(t)) >= p_drop
    keep &= ~dead[channel]
    r = t[keep] + rng.normal(0, 1, keep.sum()) * (sigma0 + nc["range_sigma_per_m"] * t[keep])
    # Directions in the level sensor frame S: world direction rotated by -yaw.
    d_sensor = d_world[keep] @ rot_z(s["yaw"])
    xyz = d_sensor * r[:, None]
    lo, hi = np.asarray(cfg["output"]["range"][:3]), np.asarray(cfg["output"]["range"][3:])
    inside = ((xyz >= lo) & (xyz <= hi)).all(axis=1)
    xyz = xyz[inside]
    if cfg["output"]["intensity"] == "return_strength":
        inten = np.clip(strength[keep][inside] * 20.0, 0, 1)
    else:
        inten = np.zeros(len(xyz))
    xyzi = np.column_stack([xyz, inten]).astype("<f4")
    return {"xyzi": xyzi, "labels": label[keep][inside].astype(np.int32),
            "params": {"beam_preset": preset["name"], "channels": int(len(elev)),
                       "azimuth_steps": int(preset.get("azimuth_steps", 1024)), "rays": int(len(t)),
                       "range_sigma_m": sigma0, "random_dropout": p_drop, "dead_channels": int(dead.sum()),
                       "returns_before_crop": int(keep.sum()), "points": int(len(xyz))}}
