"""Replay final postprocessing on screened development prediction caches."""

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "codex_review"))
from replay import calib, ev, fusion, iou, matches, read, screened

CLASSES = ("vehicle", "pedestrian", "truck")


def voter_counts(dataset, tag):
    summary = (dataset / "eval_summary.txt").read_text().splitlines()
    rows = [row for row in summary if row.startswith(tag + " | ")]
    spec = rows[-1].split(" | ")[1].removesuffix(" norule")
    totals = Counter()
    model_counts = {}
    for voter in spec.split():
        model, classes = voter.split(":")
        if model not in model_counts:
            cache = read(dataset / "allpred" / (model + "_allpred.json"))
            model_counts[model] = len(cache)
            del cache
            gc.collect()
        n = model_counts[model]
        totals.update({"all": n, **dict.fromkeys(classes.split(","), n)})
    return totals


def rescale(pred, factors):
    return {
        frame: [
            {**d, "score": min(1., d["score"] * factors.get(d["class"], 1.))}
            for d in records
        ]
        for frame, records in pred.items()
    }


def finalise(pred, hedge_floor=.02):
    rule = dict(calib.TRUCK, a=.5, base=.1)
    ruled = calib.truck_dedupe(calib.truck_rule(pred, **rule))
    # At boost=1 all scores in [0,1] remain identical regardless of point count.
    empty = dict.fromkeys(pred, np.empty((0, 3), dtype=float))
    hedged = calib.truck_boost_hedge(ruled, empty, boost=1., smin=hedge_floor)
    return calib.hedge_pedestrians(hedged), Counter(d["class"] for records in ruled.values() for d in records)


def frame_rows(pred, gt, cls, genuine=False):
    rows = []
    for f, truth in sorted(gt.items()):
        is_excluded = lambda g: genuine and g.get("type") == "TrashCan"
        boxes = [g["box"] for g in truth if g["class"] == cls and not is_excluded(g)]
        can = [g["box"] for g in truth if is_excluded(g)]
        detections = [d for d in pred[f] if d["class"] == cls]
        if can:
            overlap = iou([d["box"] for d in detections], can)
            detections = [d for d, maximum in zip(detections, overlap.max(axis=1)) if maximum < .3]
        # The kit leaves detection order unchanged on frames lacking this class.
        if boxes:
            detections.sort(key=lambda d: -d["score"])
        mat = iou([d["box"] for d in detections], boxes)
        rows.append(([d["score"] for d in detections], matches(mat), len(boxes)))
    return rows


def ap(rows):
    values = []
    for part in (rows, rows[::2], rows[1::2]):
        by_score = np.argsort(-np.concatenate([r[0] for r in part]))
        tp = np.concatenate([r[1] for r in part])[by_score].cumsum()
        precision = tp / np.arange(1, len(tp) + 1)
        recall = tp / sum(r[2] for r in part)
        values.append(round(100 * ev._compute_ap_40(precision, recall), 6))
    return values


def summary(size, label, pred, gt, hedge_floor=.02, pedestrian=None):
    out, before = finalise(pred, hedge_floor)
    values = {
        "set": size,
        "run": label,
        "ap": {c: ap(frame_rows(out, gt, c)) for c in CLASSES if c != "pedestrian"},
        "genuine": ap(frame_rows(out, gt, "truck", genuine=True)),
        "before": before,
        "after": Counter(d["class"] for records in out.values() for d in records),
    }
    values["ap"]["pedestrian"] = (pedestrian if pedestrian is not None
                              else ap(frame_rows(out, gt, "pedestrian")))
    print(json.dumps(values), flush=True)
    return values


def no_long_low(dataset, old, n):
    """Forensic only. Isolate LL removal with v9 geometry and global scale fixed."""
    summary = (dataset / "eval_summary.txt").read_text().splitlines()
    spec = [row for row in summary if row.startswith("v9C_norule | ")][-1].split(" | ")[1]
    passes = []
    ids = sorted(old)
    empty = dict.fromkeys(ids, ())
    for voter in spec.removesuffix(" norule").split():
        model, classes = voter.split(":")
        if "truck" not in classes.split(",") or model.startswith("second_twin_ll_"):
            continue
        cache = read(dataset / "allpred" / (model + "_allpred.json"))
        passes.extend({f: [d for d in data[f] if d["class"] == "truck"] for f in ids}
                      for data in cache.values())
        del cache
        gc.collect()
    assert len(passes) < n
    fusion.C = ["truck"]
    fusion.ONE_VOTE = set()
    fusion.YAW_CIRC = {"vehicle"}
    fusion.TRUCK_SCORE = ""
    trucks = fusion.fuse(passes + [empty] * (n - len(passes)), ids)
    return {f: [d for d in old[f] if d["class"] != "truck"] + trucks[f] for f in ids}


def run(size, forensic, all_nine_only=False):
    screened()
    dataset = BASE / "coop" / f"dev{size}"
    gt = {f: [g for g in labels if g["points"] >= 5]
          for f, labels in read(dataset / "gt_coop.json").items()}
    v11, v12 = "v12base_norule", "v12p3x3_norule"
    after = read(dataset / f"pred_{v12}.json")
    if all_nine_only:
        prior = Path(__file__).with_name(f"truck_replay_{size}.log")
        records = [json.loads(row) for row in prior.read_text().splitlines()]
        den11, den12 = (records[0][key] for key in ("v11_passes", "v12_passes"))
        no_pedestrian_specialists = den12["all"] - 3 * (den12["all"] - den11["all"])
        assert no_pedestrian_specialists == 124
        current = next(row for row in records if row.get("run") == "v12")
        p12 = current["ap"]["pedestrian"]
        s = den12["all"] / no_pedestrian_specialists
        summary(size, "remove_P1_P2_P3_hedge_eligibility_only", after, gt,
                .02 / s, pedestrian=p12)
        return
    den11, den12 = (voter_counts(dataset, label) for label in (v11, v12))
    print(json.dumps({"set": size, "v11_passes": den11, "v12_passes": den12}), flush=True)
    before = read(dataset / f"pred_{v11}.json")
    assert set(gt) == set(before) == set(after)
    for f in gt:
        for cls in ("vehicle", "truck"):
            orig = [d for d in before[f] if d["class"] == cls]
            actual = [d for d in after[f] if d["class"] == cls]
            assert [d["box"] for d in orig] == [d["box"] for d in actual]
            assert max((d["score"] for d in actual), default=0) < 1
            np.testing.assert_allclose(
                [d["score"] for d in orig],
                [min(1., d["score"] * den12["all"] / den11["all"]) for d in actual],
                rtol=1e-14, atol=0,
            )
    summary(size, "v11", before, gt)
    v12_stats = summary(size, "v12", after, gt)
    p12 = v12_stats["ap"]["pedestrian"]
    # Keep P3 geometry and pedestrian scores, but restore pre-P3 vehicle/truck values byte-for-byte.
    hold = {f: [d for d in after[f] if d["class"] == "pedestrian"]
                + [d for d in before[f] if d["class"] != "pedestrian"] for f in gt}
    summary(size, "freeze_vehicle_truck_at_v11", hold, gt, pedestrian=p12)
    summary(size, "v11_equivalent_truck_hedge_floor", after, gt,
            .02 * den11["all"] / den12["all"], pedestrian=p12)
    factors = {c: den12["all"] / den12[c] for c in ("vehicle", "truck")}
    summary(size, "vehicle_truck_class_denominator_only", rescale(after, factors), gt, pedestrian=p12)
    del before, after, hold
    if not forensic:
        return
    old_tag = "v9C_norule"
    old = read(dataset / f"pred_{old_tag}.json")
    old_stats = summary(size, "v9_vote_list_common_postproc", old, gt)
    fixed = no_long_low(dataset, old, voter_counts(dataset, old_tag)["all"])
    summary(size, "v9_vote_list_drop_LL_held_scale", fixed, gt, pedestrian=old_stats["ap"]["pedestrian"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", type=int, choices=(150, 350))
    parser.add_argument("--forensic", action="store_true")
    parser.add_argument("--all-nine-only", action="store_true")
    args = parser.parse_args()
    run(args.size, args.forensic, args.all_nine_only)
