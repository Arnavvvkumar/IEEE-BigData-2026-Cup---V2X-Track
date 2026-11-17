#!/bin/bash
# P9 (laptop only): wait for Lane R (fourth fine-tune), infer (BATCH=1), score it on top of v5 as a vehicle voter and as a
# pedestrian+vehicle voter, under the v5 truck rule (a 0.5, base 0.1). Pedestrian AP is on the norule line.
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
until grep -q -E 'LANER_DONE|NO_INIT_CKPT' "$base/canon/laneR.log" 2>/dev/null; do sleep 60; done
echo "P9_START $(date -Is)"
BATCH=1 bash "$base/final/run_infer_one.sh" vnx010_pv_ft4 vnx010_all.yaml claude_canon_all_twin_pv_v1 4
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V5="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian vnx010_twin_s2_ep12:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck vnx010_pv_ft_ep5:vehicle second_twin_pv_s1_ep14:truck vnx010_pv_ft2_ep5:pedestrian,vehicle"
for s in dev150 dev350; do
  echo "=== $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s v7v_norule $V5 vnx010_pv_ft4_ep4:vehicle norule
  "$P" rule_retune.py $s v7v_norule 2 'a 0.5 base 0.1'
  # shellcheck disable=SC2086
  "$P" eval150.py $s v7pv_norule $V5 vnx010_pv_ft4_ep4:pedestrian,vehicle norule
  "$P" rule_retune.py $s v7pv_norule 2 'a 0.5 base 0.1'
done
echo "P9_DONE $(date -Is)"
