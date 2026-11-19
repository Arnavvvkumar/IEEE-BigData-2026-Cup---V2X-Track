#!/bin/bash
# Full-pipeline check of the averaged heading (per class) plus truck de-duplication, on dev150 and dev350.
cd "${BDC_ROOT}/experiments/claude_carla_overnight/coop"
P="${BDC_PYTHON}"
V3="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian vnx010_twin_s2_ep12:pedestrian"
T="canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck"
for s in dev150 dev350; do
  echo "=== $s $(date -Is)"
  # shellcheck disable=SC2086
  FUSE_YAW_CIRC=vehicle "$P" eval150.py $s y_veh $V3 $T dedupe
  # shellcheck disable=SC2086
  FUSE_YAW_CIRC=vehicle,truck "$P" eval150.py $s y_vehtruck $V3 $T dedupe
  # shellcheck disable=SC2086
  FUSE_YAW_CIRC=vehicle,truck,pedestrian "$P" eval150.py $s y_all $V3 $T dedupe
done
echo YAW_EVAL_DONE
