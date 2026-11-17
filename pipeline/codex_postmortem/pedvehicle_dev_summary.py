"""Measure class mix and time coverage on screened development labels."""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import statistics


EXPERIMENT = Path(__file__).resolve().parent.parent
ROOT = EXPERIMENT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    mapping = ROOT / "experiments/real_reference_audit/v2x_real_forbidden_frames_mapping.csv"
    with mapping.open(encoding="utf-8-sig", newline="") as handle:
        forbidden = {row["point_cloud_file"].removesuffix(".bin") for row in csv.DictReader(handle)}
    assert len(forbidden) == 100
    seen = set()
    scenario_sets = []
    for size in (150, 350):
        dev = EXPERIMENT / f"coop/dev{size}"
        source = read(dev / "index.json")
        labels = read(dev / "gt_coop.json")
        keys = {row["source_key"] for row in source.values()}
        assert len(keys) == size and set(source) == set(labels)
        assert not (keys & (forbidden | seen))
        seen |= keys
        by_scenario = defaultdict(list)
        for row in source.values():
            by_scenario[row["scenario"]].append(int(row["source_key"].rsplit("/", 1)[1]))
        scenario_sets.append(set(by_scenario))
        spacings = [
            b - a for frames in by_scenario.values()
            for a, b in zip(sorted(frames), sorted(frames)[1:])
        ]
        output = {
            "frames": size,
            "scenarios": len(by_scenario),
            "median_within_scenario_frame_spacing": statistics.median(spacings),
            "labels": {},
        }
        for cls in ("vehicle", "pedestrian", "truck"):
            cohort = [row for rows in labels.values() for row in rows if row["class"] == cls and row["points"] >= 5]
            distances = [math.hypot(*row["box"][:2]) for row in cohort]
            output["labels"][cls] = {
                "count": len(cohort),
                "per_frame": round(len(cohort) / size, 2),
                "types": Counter(row["type"] for row in cohort),
                "points_5_to_15": sum(row["points"] <= 15 for row in cohort),
                "range_over_35m": sum(distance > 35 for distance in distances),
                "median_points": statistics.median(row["points"] for row in cohort),
            }
        print(json.dumps(output))
    print(json.dumps({"shared_scenarios": len(set.intersection(*scenario_sets))}))


if __name__ == "__main__":
    main()
