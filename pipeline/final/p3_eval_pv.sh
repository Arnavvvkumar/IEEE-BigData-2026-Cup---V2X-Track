#!/bin/bash
# P3: wait for Lane N (size-aware pedestrian fine-tune), run its inference frame by frame (BATCH=1), then score it in the ensemble.
# Baseline to beat = st_truckonly (dev150 69.0, dev350 68.23; pedestrian 38.7 / 37.3). One eval at a time (11 GB RAM).
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
echo "P3_WAIT $(date -Is)"
until grep -q -E 'LANEN_DONE|NO_INIT_CKPT' "$base/canon/laneN.log" 2>/dev/null; do sleep 60; done
echo "P3_START $(date -Is)"
BATCH=1 bash "$base/final/run_infer_one.sh" vnx010_pv_ft vnx010_all.yaml claude_canon_all_twin_pv_v1 5
cd "$base/coop"
"$P" ped_size_stats.py dev150 vnx010_pv_ft_ep5 vnx010_twin_s1_ep12
export FUSE_YAW_CIRC=vehicle
V3="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian vnx010_twin_s2_ep12:pedestrian"
VV="vnx010_all_s1234_ep10:vehicle vnx010_twin_s1_ep12:vehicle"
T="canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck"
for s in dev150 dev350; do
  echo "=== eval $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s pv_add  $V3 $T vnx010_pv_ft_ep5:pedestrian dedupe
  # shellcheck disable=SC2086
  "$P" eval150.py $s pv_only $VV $T vnx010_pv_ft_ep5:pedestrian dedupe
  # shellcheck disable=SC2086
  "$P" eval150.py $s pv_addv $V3 $T vnx010_pv_ft_ep5:pedestrian,vehicle dedupe
done
echo "P3_DONE $(date -Is)"
