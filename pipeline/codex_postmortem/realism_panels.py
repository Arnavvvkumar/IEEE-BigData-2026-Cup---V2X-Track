"""Audit fixed synthetic submissions on preregistered, screened dev panels."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
REF = ROOT / "experiments/real_reference_audit"
KIT = ROOT / "starting_kit/track_b"
sys.path[:0] = [str(KIT / "scoring_program"), str(BASE / "erroranalysis")]
from groundplane import fit_ground
import metrics

CONFIG = json.loads((KIT / "reference_data_dummy/config.json").read_text())
BOX = tuple(CONFIG["point_cloud_range"])
RADII = [0, 5, 10, 15, 20, 25, 30, 40, 57]


def source_sets():
    with (REF / "v2x_real_forbidden_frames_mapping.csv").open(newline="") as handle:
        ban = {row["forbidden_id"]: row["point_cloud_file"].removesuffix(".bin")
               for row in csv.DictReader(handle)}
    assert len(ban) == 100
    assert set(ban) == {row for row in (KIT / "forbidden_frames.txt").read_text().splitlines()
                        if row and not row.startswith("#")}
    seen = set()
    refs = {"legacy": (REF / "real_dev_cw1", "source_index.json", "detection_test_frames")}
    refs.update({f"dev{n}": (BASE / f"coop/dev{n}", "index.json", "scans") for n in (150, 350)})
    for name, (root, manifest, subdir) in refs.items():
        picks = {}
        for alias, row in sorted(json.loads((root / manifest).read_text()).items()):
            key = row["source_key"]
            assert key not in seen and key not in ban.values() and "/-1/" in key, key
            seen.add(key)
            picks[key] = root / subdir / f"{alias}.bin"
        yield name, picks


def balanced(rows, seed):
    # Equal acquisition-clip representation up to one frame per clip.
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for name in sorted(rows):
        groups[name.split("/")[0]].append(name)
    if len(rows) < 50:
        raise ValueError("expected at least 50 source frames")
    groups = {key: list(rng.permutation(values)) for key, values in groups.items()}
    picks = []
    while len(picks) < 50:
        for key in rng.permutation(sorted(groups)):
            if groups[key]:
                picks.append(groups[key].pop())
                if len(picks) == 50:
                    return dict((key, rows[key]) for key in sorted(picks))
    raise ValueError("expected 50 frames")


def read_scan(path):
    scan = np.fromfile(path, dtype="<f4").reshape(-1, 3)
    assert np.isfinite(scan).all(), path
    return scan


def summary(clouds):
    pooled = metrics._concat_and_mask(clouds, BOX)
    zeros = (pooled == 0).all(axis=1)
    points = pooled[~zeros].astype(float)
    radii = np.hypot(points[:, 0], points[:, 1])
    ranges = np.histogram(radii, bins=RADII)[0] / len(points)
    planes = np.asarray([fit_ground(scan, np.random.default_rng(1234))
                         for scan in clouds[::max(1, len(clouds) // 10)]])
    slope = np.linalg.norm(planes[:, :2], axis=1)
    return {
        "raw_points": sum(map(len, clouds)), "in_range": len(pooled),
        "origins": int(zeros.sum()), "origin_share": float(zeros.mean()),
        "nonzero_range_shares": ranges.round(4).tolist(),
        "nonzero_xyz_mean": points.mean(axis=0).round(4).tolist(),
        "nonzero_xyz_std": points.std(axis=0).round(4).tolist(),
        "ground_height_median": float(np.median(-planes[:, 2] / np.sqrt(1 + slope**2))),
        "ground_tilt_median_degrees": float(np.median(np.rad2deg(np.arctan(slope)))),
        "plane_scans": len(planes),
    }


def paired_streams(versions, rows):
    """Hold the reference draw fixed to isolate candidate-dependent RNG effects."""
    pools = {v: metrics._concat_and_mask(clouds, BOX) for v, clouds in versions.items()}
    reference = metrics._concat_and_mask([read_scan(path) for path in rows.values()], BOX)
    for seed in (1234, 1, 2, 3, 4):
        sampled = {}
        for v, pool in pools.items():
            rng = np.random.default_rng(seed)
            synthetic_ids = rng.choice(len(pool), metrics.N_SUBSAMPLE, replace=False)
            real_ids = rng.choice(len(reference), metrics.N_SUBSAMPLE, replace=False)
            sampled[v] = (pool[synthetic_ids], real_ids)
        original_ids = sampled[3][1]
        overlap = len(np.intersect1d(original_ids, sampled[4][1]))
        for v, (points, real_ids) in sampled.items():
            fixed_fpd = metrics.compute_fpd(points, reference[original_ids])
            official_fpd = metrics.compute_fpd(points, reference[real_ids])
            print(json.dumps({"seed": seed, "candidate": v, "ref_indices_overlap": overlap,
                              "same_reference_fpd": fixed_fpd, "kit_order_fpd": official_fpd}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("stats", "score", "streams"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234])
    parser.add_argument("--panel", action="append", help="Evaluate only these named panels")
    parser.add_argument("--count-matched", action="store_true",
                        help="Sensitivity check with 50,000 in-mask rows per real frame")
    args = parser.parse_args()
    source = dict(source_sets())
    all_dev = source["dev150"] | source["dev350"]
    panels = {"legacy": source["legacy"],
              "dev150_balanced": balanced(source["dev150"], 404),
              "dev350_balanced": balanced(source["dev350"], 405)}
    for day in sorted({key[:10] for key in all_dev}):
        available = {key: val for key, val in all_dev.items() if key.startswith(day)}
        if len(available) >= 50:
            panels[day] = balanced(available, 406)
    versions = {v: [read_scan(p) for p in sorted((BASE / f"twin/final_synthetic_v{v}/synthetic").glob("*.bin"))]
                for v in (3, 4)}
    assert all(len(clouds) == 50 for clouds in versions.values())
    if args.stage == "streams":
        paired_streams(versions, panels["legacy"])
        return
    if args.stage == "stats":
        for v, scans in versions.items():
            print(json.dumps({"candidate": v, **summary(scans)}), flush=True)
    if args.panel:
        panels = {name: panels[name] for name in args.panel}
    for name, rows in panels.items():
        assert len(rows) == 50
        ref = [read_scan(path) for path in rows.values()]
        if args.stage == "stats":
            print(json.dumps({"panel": name, "days": dict(Counter(key[:10] for key in rows)),
                              "clips": len({key.split("/")[0] for key in rows}),
                              **summary(ref)}), flush=True)
            continue
        if args.count_matched:
            rng = np.random.default_rng(830)
            ref = [metrics._mask_to_range(scan, BOX) for scan in ref]
            ref = [scan[np.sort(rng.choice(len(scan), 50_000, replace=len(scan) < 50_000))]
                   for scan in ref]
        for seed in args.seeds:
            for v, clouds in versions.items():
                raw = metrics.compute_realism(clouds, ref, BOX, seed=seed)
                scores = {k: float(np.clip((hi - raw[k]) / (hi - lo), 0, 1))
                          for k, (lo, hi) in CONFIG["normalization"].items() if k in raw}
                print(json.dumps({"panel": name, "seed": seed, "candidate": v,
                                  "count_matched": args.count_matched,
                                  "raw": raw, "normalized": scores,
                                  "score": float(np.mean(list(scores.values())))}), flush=True)


if __name__ == "__main__":
    main()
