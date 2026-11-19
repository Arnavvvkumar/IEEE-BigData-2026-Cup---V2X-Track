#!/bin/bash
# Priority 1: add second_twin_s2 (SECOND, merged twin set, ep12) to the v3 ensemble.
# Step 0 checks that the batched inference (Cursor's batched_forward) equals batch-1 inference on the same frames.
# Step 1 runs the model on dev150, the 50 test scans and dev350 (no labels). Step 2 scores ensembles on dev150 and dev350.
set -uo pipefail
root="${BDC_ROOT}"
base="$root/experiments/claude_carla_overnight"
here="$base/final"
program="${BDC_PROGRAM}"
repo="$program/OpenPCDet"
P="$program/env310/bin/python"
D="${BDC_DATA}"
tmp="$base/coop/batchcheck"
mkdir -p "$tmp"
echo "P1_START $(date -Is)"
ck=$(dirname "$(find "$repo/output" -path "*/second_twin_s1/ckpt/checkpoint_epoch_12.pth" -print -quit)")
for b in 1 4; do
  ( cd "$repo/tools" && "$P" "$here/infer_test.py" --repo "$repo" --cfg "$base/canon/second_canon_all_v1.yaml" --data "$D/claude_canon_all_twin_v1" \
      --checkpoint "$ck/checkpoint_epoch_12.pth" --scans "$base/coop/dev150/scans" --out "$tmp/b$b.json" --ground -3.0 -2.85 --limit 16 --batch $b --workers 8 2>&1 \
      | grep -E '^scans|^ground|Error|Traceback' | sed "s/^/batch$b /" )
  sleep 10
done
"$P" - << 'PY'
import json
from pathlib import Path
t = Path(__import__('os').environ['BDC_ROOT'] + "/experiments/claude_carla_overnight/coop/batchcheck")
a, b = json.loads((t / "b1.json").read_text()), json.loads((t / "b4.json").read_text())
worst, ndiff = 0.0, 0
for g in a:
    for f in a[g]:
        x, y = a[g][f], b[g][f]
        if len(x) != len(y):
            ndiff += 1
            continue
        for p, q in zip(x, y):
            worst = max(worst, abs(p["score"] - q["score"]), max(abs(u - v) for u, v in zip(p["box"], q["box"])))
print("BATCHCHECK frames_with_count_diff", ndiff, "max_abs_diff", worst)
PY
echo "=== infer second_twin_s2 ep12 $(date -Is)"
bash "$here/run_infer_one.sh" second_twin_s2 second_canon_all_v1.yaml claude_canon_all_twin_v1 12
cd "$base/coop"
V3="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian"
for s in dev150 dev350; do
  echo "=== eval $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s p1_v3      $V3 canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck
  # shellcheck disable=SC2086
  "$P" eval150.py $s p1_addT2   $V3 canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck
  # shellcheck disable=SC2086
  "$P" eval150.py $s p1_T2forB  $V3 canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck
  # shellcheck disable=SC2086
  "$P" eval150.py $s p1_twinsOnly $V3 second_twin_s1_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck
done
echo "P1_DONE $(date -Is)"
