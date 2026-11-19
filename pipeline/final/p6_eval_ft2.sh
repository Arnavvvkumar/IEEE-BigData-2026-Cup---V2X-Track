#!/bin/bash
# P6: wait for Lane P (second size-aware VoxelNeXt fine-tune), infer (BATCH=1), score as vehicle voter / pedestrian voter
# on top of candidate v4 (o_truck: dev150 69.43, dev350 68.73). One eval at a time.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
until grep -q -E 'LANEP_DONE|NO_INIT_CKPT' "$base/canon/laneP.log" 2>/dev/null; do sleep 60; done
echo "P6_START $(date -Is)"
BATCH=1 bash "$base/final/run_infer_one.sh" vnx010_pv_ft2 vnx010_all.yaml claude_canon_all_twin_pv_v1 5
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V4="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian vnx010_twin_s2_ep12:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck vnx010_pv_ft_ep5:vehicle second_twin_pv_s1_ep14:truck"
for s in dev150 dev350; do
  echo "=== eval $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s ft2_veh $V4 vnx010_pv_ft2_ep5:vehicle dedupe
  # shellcheck disable=SC2086
  "$P" eval150.py $s ft2_pedveh $V4 vnx010_pv_ft2_ep5:pedestrian,vehicle dedupe
done
echo "P6_DONE $(date -Is)"
