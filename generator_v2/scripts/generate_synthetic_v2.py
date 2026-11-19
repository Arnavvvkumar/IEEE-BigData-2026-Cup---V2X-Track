"""Generate procedural roadside LiDAR scenes (generator v2), CPU only.

Method: procedural triangle meshes (lofted vehicle bodies, trucks/buses,
articulated-pose pedestrians, buildings, poles, trees, curbs) on sampled road
layouts, ray cast with Open3D's RaycastingScene (Embree) from a configurable
roadside sensor, followed by a simple measurement model (range noise, beam
jitter, return-strength/random/dead-channel dropout). Not a reconstructed
digital twin, not CARLA output, and no external assets or real data are used.

Output (compatible with the V2X-Real loader contract):
  <output>/<train|validate>/<scenario>/<agent>/<frame>.bin   float32 XYZI, sensor frame
  <output>/<train|validate>/<scenario>/<agent>/<frame>.yaml  lidar_pose, ego_speed, vehicles
  <output>/aux/point_labels/<split>/<frame>.npy              int32 per-point label (object id or <0 background)
  <output>/frames.json        per-frame manifest (center/full LWH/yaw-rad boxes in sensor frame)
  <output>/provenance.json    full config, code hashes, environment, timing (written last)

Example:
  generate_synthetic_v2.py --output /home/arnav/projects/bdc2026-track-b/data/claude_v2_smoke \
      --train 32 --validate 8 --seed 2026
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_v2 import config as C  # noqa: E402
from synthetic_v2 import geometry as G  # noqa: E402
from synthetic_v2 import layout as L  # noqa: E402
from synthetic_v2 import sensor as S  # noqa: E402
from synthetic_v2 import traffic as T  # noqa: E402

SCRIPT = Path(__file__).resolve()
MODULE_DIR = SCRIPT.parent / "synthetic_v2"
CLASSES = ("vehicle", "pedestrian", "truck")
STREAM_LAYOUT, STREAM_TRAFFIC, STREAM_MEASURE, STREAM_CALIB = 0, 1, 2, 3

METHOD = ("Procedural meshes (lofted car bodies; box/cylinder/ellipsoid trucks, buses, pedestrians, "
          "buildings, poles, trees, curbs) on sampled road layouts, ray cast on CPU with Open3D "
          "RaycastingScene (Embree) plus a simple noise/dropout model. Not a reconstructed digital twin, "
          "not CARLA output; no external assets, real clouds, labels or statistics.")
LIMITATIONS = [
    "Flat ground plane; straight road arms; no terrain, driveways, ramps or lane paint.",
    "Vehicle bodies are lofted convex cross-sections: no wheel wells, mirrors, open pickup beds or interiors; glass is opaque and dark.",
    "Trucks/buses/vans are box assemblies; semi trailers are never articulated.",
    "Pedestrians are rigid limb/ellipsoid assemblies in one walking or standing pose; no cyclists, scooters or motorcycles.",
    "Single-return sensor; no beam divergence, mixed pixels, multipath, weather or motion skew.",
    "Foliage porosity is a per-surface pass-through probability, not a leaf-level model.",
    "Intensity defaults to zero (XYZ-only); the optional return_strength mode is uncalibrated.",
    "Static scene snapshot; traffic follows lane/sidewalk rules but no signal-phase or interaction logic.",
]


def rng_for(seed, *keys):
    return np.random.default_rng(np.random.SeedSequence([int(seed)] + [int(k) for k in keys]))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    files = [SCRIPT] + sorted(MODULE_DIR.glob("*.py"))
    hashes = {str(f.relative_to(SCRIPT.parents[1])).replace("\\", "/"): sha256_file(f) for f in files}
    combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()
    return hashes, combined


def build_layout(seed, layout_id, cfg):
    rng = rng_for(seed, layout_id, STREAM_LAYOUT)
    layout = L.sample_layout(layout_id, rng, cfg)
    L.build_static(layout, rng, cfg)
    return layout


def plan_tasks(cfg, train, validate):
    fpl = int(cfg["splits"]["frames_per_layout"])
    tasks, serial = [], 0
    for split, count in (("train", train), ("validate", validate)):
        offset = int(cfg["splits"]["layout_id_offset"][split])
        for i in range(count):
            layout_index, k = divmod(i, fpl)
            layout_id = offset + layout_index
            tasks.append({"serial": serial, "split": split, "index": i, "layout_id": layout_id,
                          "frame_in_layout": k, "frame": f"v2{split[0]}_L{layout_id:07d}_F{k:02d}"})
            serial += 1
    return tasks


MAX_LAYOUTS_PER_SPLIT = 10000


def object_id(task, cfg, j):
    """Globally unique int32 object ID, independent of split sizes.

    ID = layout_ordinal * 1e5 + frame_in_layout * 1e3 + index + 1, where the
    validate ordinals start at 10000, so the largest ID stays below 2**31.
    """
    ordinal = task["layout_id"] - int(cfg["splits"]["layout_id_offset"][task["split"]])
    if not 0 <= ordinal < MAX_LAYOUTS_PER_SPLIT:
        raise ValueError(f"At most {MAX_LAYOUTS_PER_SPLIT} layouts per split are supported")
    if task["split"] == "validate":
        ordinal += MAX_LAYOUTS_PER_SPLIT
    oid = ordinal * 100000 + task["frame_in_layout"] * 1000 + j + 1
    if j >= 999 or oid >= 2 ** 31:
        raise ValueError("Object ID out of int32 range")
    return oid


def render_frame(task, cfg, seed, layout=None):
    """Simulate one frame. Returns (xyzi, point_labels, yaml_params, record)."""
    layout = layout or build_layout(seed, task["layout_id"], cfg)
    k = task["frame_in_layout"]
    users, traffic_info = T.place_traffic(layout, rng_for(seed, task["layout_id"], STREAM_TRAFFIC, k), cfg)
    scene = S.RayScene(cfg["runtime"]["raycast_threads"])
    for mesh, label, porosity in layout.static:
        scene.add(mesh, label, porosity)
    ids = []
    for j, u in enumerate(users):
        oid = object_id(task, cfg, j)
        scene.add(u.mesh.transformed(G.rot_z(u.yaw), (u.x, u.y, u.ground_z)), oid)
        ids.append(oid)
    sim = S.simulate(scene, layout, cfg, rng_for(seed, task["layout_id"], STREAM_MEASURE, k),
                     rng_for(seed, task["layout_id"], STREAM_CALIB))
    points, labels = sim["xyzi"], sim["labels"]
    sensor = layout.sensor
    sxyz, syaw = sensor["xyz"], sensor["yaw"]
    lo, hi = np.asarray(cfg["output"]["range"][:3]), np.asarray(cfg["output"]["range"][3:])
    ac = cfg["annotation"]
    world_mode = cfg["output"]["yaml_pose_mode"] == "world"
    vehicles, ground_truth, details = {}, [], []
    for u, oid in zip(users, ids):
        box_w = u.box_world
        box_s = G.world_box_to_sensor(box_w, sxyz, syaw)
        hits = int((labels == oid).sum())
        corners = G.box_corners_3d(box_s)
        in_range = int(((corners >= lo) & (corners <= hi)).all(axis=1).sum())
        annotated = hits >= ac["min_points"] and in_range >= ac["min_corners_in_range"]
        details.append({"id": oid, "class": u.cls, "obj_type": u.obj_type, "subtype": u.subtype,
                        "placement": u.placement, "box": box_s.tolist(), "box_world": box_w.tolist(),
                        "hits": hits, "corners_in_range": in_range, "annotated": bool(annotated),
                        "range_m": float(np.hypot(box_s[0], box_s[1]))})
        if annotated:
            src = box_w if world_mode else box_s
            vehicles[oid] = G.box_to_yaml(src, u.obj_type, u.speed)
            ground_truth.append({"box": box_s.tolist(), "class": u.cls})
    if world_mode:
        lidar_pose = [float(sxyz[0]), float(sxyz[1]), float(sxyz[2]), 0.0, math.degrees(syaw), 0.0]
    else:
        lidar_pose = [0.0] * 6
    params = {"lidar_pose": lidar_pose, "ego_speed": 0.0, "vehicles": vehicles}
    class_counts = Counter(g["class"] for g in ground_truth)
    record = {
        "split": task["split"], "frame": task["frame"], "serial": task["serial"],
        "scenario": cfg["output"]["scenario_dir"], "agent": str(cfg["output"]["agent_id"]),
        "layout_id": task["layout_id"], "layout_family": layout.family,
        "layout_fingerprint": layout.fingerprint(), "frame_in_layout": k,
        "sensor": {"placement": sensor["placement"], "xyz_world": [float(v) for v in sxyz],
                   "height_m": float(sxyz[2]), "roll_deg": math.degrees(sensor["roll"]),
                   "pitch_deg": math.degrees(sensor["pitch"]), "yaw_deg": math.degrees(syaw)},
        "measurement": sim["params"],
        "points": int(len(points)), "objects": len(ground_truth),
        "class_counts": {c: int(class_counts.get(c, 0)) for c in CLASSES},
        "traffic": traffic_info, "ground_truth": ground_truth, "object_details": details,
    }
    return points, labels, params, record


def frame_paths(output, cfg, task):
    agent_dir = Path(output) / task["split"] / cfg["output"]["scenario_dir"] / str(cfg["output"]["agent_id"])
    return (agent_dir / f"{task['frame']}.bin", agent_dir / f"{task['frame']}.yaml",
            Path(output) / "aux" / "point_labels" / task["split"] / f"{task['frame']}.npy")


def _render_group(args):
    """Worker entry: render all frames of one layout (layout built once)."""
    tasks, cfg, seed, output = args
    layout = build_layout(seed, tasks[0]["layout_id"], cfg)
    records, seconds = [], []
    for task in tasks:
        started = time.perf_counter()
        points, labels, params, record = render_frame(task, cfg, seed, layout)
        bin_path, yaml_path, label_path = frame_paths(output, cfg, task)
        points.tofile(bin_path)
        yaml_path.write_text(yaml.safe_dump(params, sort_keys=False))
        if cfg["output"]["write_point_labels"]:
            np.save(label_path, labels)
            record["point_labels"] = str(label_path.relative_to(output)).replace("\\", "/")
        record["path"] = str(bin_path.relative_to(output)).replace("\\", "/")
        record["point_file"] = record["path"]  # key read by scripts/run_initial_experiments.py
        record["sha256"] = sha256_file(bin_path)
        record["yaml_sha256"] = sha256_file(yaml_path)
        seconds.append(time.perf_counter() - started)
        records.append(record)
    return records, seconds


def limit_cpus(n):
    """Pin this process (and the workers it spawns) to at most n CPUs.

    Embree's BVH build uses its own thread pool, which the per-cast thread
    argument does not bound; CPU affinity does. Returns the CPU list used.
    """
    if not hasattr(os, "sched_setaffinity"):
        return None
    allowed = sorted(os.sched_getaffinity(0))
    chosen = allowed[:max(1, n)]
    os.sched_setaffinity(0, chosen)
    return chosen


def disk_usage(output):
    usage = defaultdict(int)
    for f in Path(output).rglob("*"):
        if f.is_file():
            usage[f.suffix or "other"] += f.stat().st_size
    usage["total"] = sum(usage.values())
    return dict(usage)


def generate(output, cfg, seed=2026, train=32, validate=8, workers=None, config_path=None, argv=None):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output already exists and is not an empty directory: {output}")
    workers = int(workers or cfg["runtime"]["workers"])
    if not 1 <= workers <= C.MAX_WORKERS:
        raise ValueError(f"workers must be 1..{C.MAX_WORKERS}")
    if train < 0 or validate < 0:
        raise ValueError("frame counts must be non-negative")
    # Keep total CPU use at the configured ray-cast thread budget (default 2) regardless of
    # worker count. Thread count never changes the output bytes.
    requested_threads = int(cfg["runtime"]["raycast_threads"])
    threads = max(1, requested_threads // workers)
    cfg = dict(cfg, runtime=dict(cfg["runtime"], workers=workers, raycast_threads=threads))
    C.validate(cfg)
    tasks = plan_tasks(cfg, train, validate)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validate"):
        (output / split / cfg["output"]["scenario_dir"] / str(cfg["output"]["agent_id"])).mkdir(parents=True)
        (output / "aux" / "point_labels" / split).mkdir(parents=True)
    cpus = limit_cpus(int(cfg["runtime"]["cpu_limit"]))
    groups = defaultdict(list)
    for task in tasks:
        groups[task["layout_id"]].append(task)
    jobs = [(g, cfg, seed, output) for g in groups.values()]
    started_utc = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    if workers == 1:
        results = [_render_group(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(_render_group, jobs))
    elapsed = time.perf_counter() - t0
    records = sorted((r for group, _ in results for r in group), key=lambda r: r["serial"])
    frame_seconds = [s for _, group in results for s in group]

    split_layouts = {s: sorted({r["layout_id"] for r in records if r["split"] == s}) for s in ("train", "validate")}
    split_prints = {s: {r["layout_fingerprint"] for r in records if r["split"] == s} for s in ("train", "validate")}
    if set(split_layouts["train"]) & set(split_layouts["validate"]) or split_prints["train"] & split_prints["validate"]:
        raise RuntimeError("Layout leakage between train and validate")

    (output / "frames.json").write_text(json.dumps(records, indent=1))
    hashes, combined = code_hashes()
    config_file = Path(config_path) if config_path else None
    import open3d
    provenance = {
        "source": "synthetic", "generated_by_team": True, "uses_real_data": False,
        "generator": "scripts/generate_synthetic_v2.py", "generator_version": 2, "method": METHOD,
        "limitations": LIMITATIONS,
        "code_sha256": hashes, "code_sha256_combined": combined,
        "config_path": str(config_file) if config_file else None,
        "config_file_sha256": sha256_file(config_file) if config_file and config_file.exists() else None,
        "config": cfg, "seed": seed, "train_frames": train, "validate_frames": validate, "workers": workers,
        "raycast_threads": {"requested_total": requested_threads, "per_worker": threads},
        "cpu_affinity": cpus,
        "argv": list(argv) if argv is not None else None,
        "layout_ids": split_layouts,
        "frames_json_sha256": sha256_file(output / "frames.json"),
        "coordinate_frame": ("Level sensor frame: origin at the LiDAR, x/y rotated by the mount yaw, z up "
                             "(gravity aligned). Mount roll/pitch tilt the beams only. Ground is at z = -sensor height."),
        "annotation_contract": ("YAML vehicles: location = box center, center = [0,0,0], extent = half sizes "
                                "[l/2, w/2, h/2], angle = [0, yaw_deg, 0], obj_type Car/Pedestrian/Truck. "
                                "frames.json ground_truth: box = [x, y, z, l, w, h, yaw_rad] (center, full sizes)."),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "open3d": open3d.__version__,
                        "pyyaml": yaml.__version__, "platform": platform.platform()},
        "timing": {"started_utc": started_utc, "elapsed_s": round(elapsed, 3),
                   "frames_per_s": round(len(records) / elapsed, 4) if elapsed > 0 else None,
                   "mean_frame_s": round(float(np.mean(frame_seconds)), 4) if frame_seconds else None,
                   "max_frame_s": round(float(np.max(frame_seconds)), 4) if frame_seconds else None},
        "disk_bytes": disk_usage(output),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=float))
    print(f"Generated {train} training + {validate} validation frames at {output} "
          f"in {elapsed:.1f}s ({provenance['timing']['frames_per_s']} frames/s, "
          f"{provenance['disk_bytes']['total'] / 2**20:.1f} MiB)", flush=True)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new or empty directory")
    parser.add_argument("--config", type=Path, default=C.DEFAULT_CONFIG)
    parser.add_argument("--train", type=int, default=32)
    parser.add_argument("--validate", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=None, help=f"1..{C.MAX_WORKERS} (default from config)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set traffic.empty_scene_prob=0.2")
    args = parser.parse_args(argv)
    cfg = C.load_config(args.config, args.set)
    generate(args.output, cfg, args.seed, args.train, args.validate, args.workers, args.config,
             argv=sys.argv if argv is None else ["generate_synthetic_v2.py", *argv])


if __name__ == "__main__":
    main()
