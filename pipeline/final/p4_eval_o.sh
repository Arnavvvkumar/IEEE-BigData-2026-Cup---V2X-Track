#!/bin/bash
# P4: (1) size-aware pedestrian model used as a VEHICLE voter only; (2) wait for Lane O (SECOND on the full merged set), infer, score.
# Baseline = st_truckonly (dev150 69.0, dev350 68.23). One eval at a time (11 GB RAM).
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V3="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian vnx010_twin_s2_ep12:pedestrian"
T="canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck"
PV="vnx010_pv_ft_ep5:vehicle"
echo "P4_START $(date -Is)"
for s in dev150 dev350; do
  # shellcheck disable=SC2086
  "$P" eval150.py $s pv_veh $V3 $T $PV dedupe
done
until grep -q -E 'LANEO_DONE' "$base/canon/laneO.log" 2>/dev/null; do sleep 60; done
echo "P4_INFER $(date -Is)"
bash "$base/final/run_infer_one.sh" second_twin_pv_s1 second_canon_all_v1.yaml claude_canon_all_twin_pv_v1 14
cd "$base/coop"
O=second_twin_pv_s1_ep14
for s in dev150 dev350; do
  echo "=== eval $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s o_add   $V3 $T $PV $O:vehicle,truck dedupe
  # shellcheck disable=SC2086
  "$P" eval150.py $s o_truck $V3 $T $PV $O:truck dedupe
  # shellcheck disable=SC2086
  "$P" eval150.py $s o_veh   $V3 $T $PV $O:vehicle dedupe
done
echo "P4_DONE $(date -Is)"
