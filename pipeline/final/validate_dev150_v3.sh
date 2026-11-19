#!/bin/bash
# Same fuse -> static -> calibrate -> truck_rule path as candidate v3, on dev150.
# Confirms the AP of coop/dev150/pred_j_ttaBothPed.json.
set -euo pipefail
root="${BDC_ROOT}"
here="$root/experiments/claude_carla_overnight/final"
coop="$root/experiments/claude_carla_overnight/coop"
P="${BDC_PYTHON}"
A="$coop/dev150/allpred"
scans="$coop/dev150/scans"
fused="$coop/dev150/pred_v3_pipeline_fused.json"
pred="$coop/dev150/pred_v3_pipeline.json"
"$P" "$here/check_forbidden.py" "$coop/dev150"
"$P" "$here/fuse_test.py" "$scans" "$fused" \
  "$A/vnx010_all_s1234_ep10_allpred.json::pedestrian,vehicle" \
  "$A/vnx010_all_rs2_amp_ep12_allpred.json::pedestrian" \
  "$A/vnx010_twin_s1_ep12_allpred.json::pedestrian,vehicle" \
  "$A/canon_all_s1234_ep12_allpred.json::vehicle,truck" \
  "$A/second_twin_s1_ep12_allpred.json::vehicle,truck" \
  "$A/canon_relabel_s1234_ep12_allpred.json::vehicle,truck" \
  "$A/vnx010_twin_s1_tta_ep12_allpred.json::pedestrian" \
  "$A/vnx010_all_s1234_tta_ep10_allpred.json::pedestrian"
"$P" "$here/calib.py" "$scans" "$fused" "$pred"
echo "pipeline $( "$P" "$here/score_coop.py" "$coop/dev150" "$pred" )"
echo "reference $( "$P" "$here/score_coop.py" "$coop/dev150" "$coop/dev150/pred_j_ttaBothPed.json" )"
"$P" - << 'PY'
import json
from pathlib import Path
coop = Path(__import__('os').environ['BDC_ROOT'] + "/experiments/claude_carla_overnight/coop/dev150")
a = json.loads((coop / "pred_v3_pipeline.json").read_text())
b = json.loads((coop / "pred_j_ttaBothPed.json").read_text())
print("frames", len(a), len(b), "same_ids", set(a) == set(b))
n = 0
max_box = 0.0
max_score = 0.0
for f in a:
    da, db = a[f], b[f]
    if len(da) != len(db):
        n += 1
        continue
    for x, y in zip(da, db):
        if x["class"] != y["class"]:
            n += 1
            break
        max_score = max(max_score, abs(x["score"] - y["score"]))
        max_box = max(max_box, max(abs(p - q) for p, q in zip(x["box"], y["box"])))
print("frames_with_count_or_class_diff", n, "max_score_diff", max_score, "max_box_diff", max_box)
stamp = Path(__import__('os').environ['BDC_ROOT'] + "/experiments/claude_carla_overnight/final/dev150_v3_ok")
if n == 0 and max_score < 1e-6 and max_box < 1e-6:
    stamp.write_text("match\n")
    print("DEV150_MATCH")
else:
    stamp.write_text("diff\n")
    raise SystemExit(2)
PY
echo DEV150_PIPELINE_DONE
