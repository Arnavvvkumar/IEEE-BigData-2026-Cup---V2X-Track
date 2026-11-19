"""Configuration loading, dotted overrides and validation."""
import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "generator_v2.yaml"
# Upper bound only; the shipped config still defaults to 1 worker on 2 CPUs.
# Worker/thread counts never change output bytes (tested).
MAX_WORKERS = max(1, os.cpu_count() or 1)


def load_config(path=None, overrides=None):
    path = Path(path or DEFAULT_CONFIG)
    cfg = yaml.safe_load(path.read_text())
    for item in overrides or []:
        apply_override(cfg, item)
    validate(cfg)
    return cfg


def apply_override(cfg, item):
    """Apply 'a.b.c=<yaml value>' (or a (key, value) pair) to a nested dict."""
    if isinstance(item, str):
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key=value: {item!r}")
        value = yaml.safe_load(raw)
    else:
        key, value = item
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"Unknown config section in override: {key}")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"Unknown config key in override: {key}")
    node[parts[-1]] = copy.deepcopy(value)


def _range(cfg, *keys, lo=None, hi=None):
    node = cfg
    for k in keys:
        node = node[k]
    if not (isinstance(node, (list, tuple)) and len(node) == 2 and node[0] <= node[1]):
        raise ValueError(f"{'.'.join(keys)} must be [min, max] with min <= max, got {node}")
    if lo is not None and node[0] < lo or hi is not None and node[1] > hi:
        raise ValueError(f"{'.'.join(keys)} must lie within [{lo}, {hi}], got {node}")


def validate(cfg):
    if cfg.get("version") != 2:
        raise ValueError("generator_v2 config must have version: 2")
    out = cfg["output"]
    r = out["range"]
    if len(r) != 6 or not all(r[i] < r[i + 3] for i in range(3)):
        raise ValueError(f"output.range must be [xmin, ymin, zmin, xmax, ymax, zmax], got {r}")
    if int(out["agent_id"]) >= 0:
        raise ValueError("output.agent_id must be negative (infrastructure)")
    if out["yaml_pose_mode"] not in ("sensor", "world"):
        raise ValueError("output.yaml_pose_mode must be sensor or world")
    if out["intensity"] not in ("zero", "return_strength"):
        raise ValueError("output.intensity must be zero or return_strength")
    if not 1 <= cfg["splits"]["frames_per_layout"] <= 99:
        raise ValueError("splits.frames_per_layout must be 1..99 (object IDs encode it in two digits)")
    off = cfg["splits"]["layout_id_offset"]
    if off["train"] == off["validate"]:
        raise ValueError("train and validate layout ID offsets must differ")
    w = int(cfg["runtime"]["workers"])
    if not 1 <= w <= MAX_WORKERS:
        raise ValueError(f"runtime.workers must be 1..{MAX_WORKERS}")
    if int(cfg["runtime"]["raycast_threads"]) < 1:
        raise ValueError("runtime.raycast_threads must be >= 1")
    if not 1 <= int(cfg["runtime"].get("cpu_limit", 0)) <= MAX_WORKERS:
        raise ValueError(f"runtime.cpu_limit must be 1..{MAX_WORKERS}")
    for key in ("lane_width_m", "median_width_m", "parking_width_m", "sidewalk_width_m", "curb_height_m"):
        _range(cfg, "layout", key, lo=0)
    if abs(sum(cfg["layout"]["family_weights"].values())) <= 0:
        raise ValueError("layout.family_weights must have positive mass")
    _range(cfg, "sensor", "height_m", lo=0.5)
    zmin = r[2]
    if -cfg["sensor"]["height_m"][1] < zmin:
        raise ValueError("sensor.height_m max puts the ground below output.range zmin")
    for preset in cfg["sensor"]["beam_presets"]:
        if preset["pattern"] not in ("uniform", "gradient", "explicit"):
            raise ValueError(f"Unknown beam pattern in preset {preset['name']}")
        if preset["pattern"] == "explicit":
            if not preset.get("elevations_deg"):
                raise ValueError(f"explicit preset {preset['name']} needs elevations_deg")
        else:
            if preset["channels"] < 1 or preset["azimuth_steps"] < 8:
                raise ValueError(f"preset {preset['name']} needs channels >= 1 and azimuth_steps >= 8")
            _range(preset, "elevation_deg", lo=-90, hi=90)
    if not 0 <= cfg["sensor"]["min_range_m"] < cfg["sensor"]["max_range_m"]:
        raise ValueError("sensor.min_range_m must be < max_range_m")
    _range(cfg, "noise", "range_sigma_m", lo=0)
    _range(cfg, "dropout", "random_prob", lo=0, hi=1)
    tr = cfg["traffic"]
    for cls in ("vehicle", "pedestrian", "truck"):
        lo, hi = tr["count_range"][cls]
        if not 0 <= lo <= hi:
            raise ValueError(f"traffic.count_range.{cls} must be 0 <= min <= max")
        if not 0 <= tr["class_absent_prob"][cls] <= 1:
            raise ValueError(f"traffic.class_absent_prob.{cls} must be a probability")
    if sum(tr["count_range"][c][1] for c in ("vehicle", "pedestrian", "truck")) > 999:
        raise ValueError("traffic.count_range maxima must sum to <= 999 (object IDs use three digits)")
    if not 0 <= tr["empty_scene_prob"] <= 1:
        raise ValueError("traffic.empty_scene_prob must be a probability")
    if cfg["annotation"]["min_points"] < 1:
        raise ValueError("annotation.min_points must be >= 1")
    return cfg
