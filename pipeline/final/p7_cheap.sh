#!/bin/bash
# P7 cheap post-processing tests on the v5 ensemble (v4 + second size-aware model for pedestrians and vehicles):
#   1) calibrated predictions without the truck rule  2) truck-rule retune  3) rider-sized pedestrian boxes.
# One job at a time (11 GB RAM). dev150 first (fast), then dev350.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
cd "$base/coop"
while pgrep -f truck_h_test.py > /dev/null; do sleep 20; done
export FUSE_YAW_CIRC=vehicle
V5="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian vnx010_twin_s2_ep12:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck vnx010_pv_ft_ep5:vehicle second_twin_pv_s1_ep14:truck vnx010_pv_ft2_ep5:pedestrian,vehicle"
for s in dev150 dev350; do
  echo "=== $s norule $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s v5_norule $V5 norule
  echo "=== $s rule retune $(date -Is)"
  "$P" rule_retune.py $s v5_norule
  echo "=== $s riders $(date -Is)"
  "$P" rider_test.py $s ft2_pedveh
done
echo "P7_DONE $(date -Is)"
