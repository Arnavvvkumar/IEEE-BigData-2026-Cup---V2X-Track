#!/bin/bash
# P4: add vnx010_twin_s2 (fine-grid VoxelNeXt, second seed, merged twin set) to the ensemble for pedestrians/vehicles.
# Inference is frame-by-frame (BATCH=1) because batched pedestrian boxes differ slightly from batch-1. Shares the GPU with training (small).
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
echo "P4_START $(date -Is)"
BATCH=1 bash "$base/final/run_infer_one.sh" vnx010_twin_s2 vnx010_all.yaml claude_canon_all_twin_v1 12
cd "$base/coop"
V3="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian"
T="canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck"
for s in dev150 dev350; do
  echo "=== eval $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s p4_base    $V3 $T
  # shellcheck disable=SC2086
  "$P" eval150.py $s p4_addV2ped $V3 vnx010_twin_s2_ep12:pedestrian $T
  # shellcheck disable=SC2086
  "$P" eval150.py $s p4_addV2both $V3 vnx010_twin_s2_ep12:pedestrian,vehicle $T
done
echo "P4_DONE $(date -Is)"
