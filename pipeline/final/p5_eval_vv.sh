#!/bin/bash
# P5: after P4, test existing pedestrian-only VoxelNeXt models as extra VEHICLE voters (no new inference needed).
# Baseline = pv_veh (dev150 69.17, dev350 68.63; vehicle 80.7 / 81.2).
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
until grep -q 'P4_DONE' "$base/final/p4_eval_o.log" 2>/dev/null; do sleep 60; done
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
T="canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck"
PV="vnx010_pv_ft_ep5:vehicle"
VA="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian vnx010_twin_s2_ep12:pedestrian,vehicle"
VB="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian vnx010_twin_s2_ep12:pedestrian,vehicle"
echo "P5_START $(date -Is)"
for s in dev150 dev350; do
  echo "=== eval $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s vv_s2    $VA $T $PV dedupe
  # shellcheck disable=SC2086
  "$P" eval150.py $s vv_s2rs2 $VB $T $PV dedupe
done
echo "P5_DONE $(date -Is)"
