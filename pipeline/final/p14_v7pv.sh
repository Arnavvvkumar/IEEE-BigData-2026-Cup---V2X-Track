#!/bin/bash
# Rebuild v7 with PV-RCNN epoch 2 as pedestrian + vehicle voter, then confirm the combined config on both validation sets.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
cp "$base/final/candidate_v7/predictions.json" "$base/final/candidate_v7_meta/predictions_without_pvrcnn.json"
EXTRA="pvrcnn_st_osc_ep2:pedestrian,vehicle" bash "$base/final/build_candidate_v7.sh" candidate_v7 bdc2026_trackB_submission_v7.zip 1
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V7="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle"
for s in dev150 dev350; do
  # shellcheck disable=SC2086
  "$P" eval150.py $s v7pv_norule $V7 pvrcnn_st_osc_ep2:pedestrian,vehicle norule > /dev/null
  "$P" verify_hedge_fn.py $s v7pv_norule | tail -n 1
  "$P" rule_retune.py $s v7pv_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s v7+pvrcnn: /"
done
echo P14_DONE
