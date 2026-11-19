#!/bin/bash
# Shift/rotation passes of the pedestrian models on the 150 validation frames, then pedestrian comparison.
base="${BDC_ROOT}/experiments/claude_carla_overnight"
cd "$base/coop"
while pgrep -f 'infer_test.py' > /dev/null; do sleep 20; done
sleep 30
bash "$base/final/run_tta_one.sh" vnx010_twin_s1 vnx010_all.yaml claude_canon_all_twin_v1 12 "$base/coop/dev150/scans" "$base/coop/dev150/allpred"
sleep 20
bash "$base/final/run_tta_one.sh" vnx010_all_s1234 vnx010_all.yaml claude_canon_all_v1 10 "$base/coop/dev150/scans" "$base/coop/dev150/allpred"
echo TTA_DEV150_DONE
