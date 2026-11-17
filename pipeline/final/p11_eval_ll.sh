#!/bin/bash
# P11 (laptop): score the long-low-truck SECOND (second_twin_ll_s1 ep12) on top of the v6 spec, under the v5 truck rule (a 0.5, base 0.1).
# Variants: truck voter only; vehicle+truck voter; truck voter together with the fourth fine-tune as a vehicle voter (the v7 candidate).
# One eval at a time. Baseline v6: dev150 vehicle 81.6 truck 89.5 genuine 74.9 | dev350 81.5 / 87.6 / 72.3.
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
echo "P11_START $(date -Is)"
bash "$base/final/run_infer_one.sh" second_twin_ll_s1 second_canon_all_v1.yaml claude_canon_all_twin_ll_v1 12
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V6="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck"
LL=second_twin_ll_s1_ep12
for s in dev150 dev350; do
  echo "=== $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s ll_t_norule $V6 $LL:truck norule
  "$P" rule_retune.py $s ll_t_norule 2 'a 0.5 base 0.1'
  # shellcheck disable=SC2086
  "$P" eval150.py $s ll_vt_norule $V6 $LL:vehicle,truck norule
  "$P" rule_retune.py $s ll_vt_norule 2 'a 0.5 base 0.1'
  # shellcheck disable=SC2086
  "$P" eval150.py $s v7_norule $V6 $LL:truck vnx010_pv_ft4_ep4:vehicle norule
  "$P" rule_retune.py $s v7_norule 2 'a 0.5 base 0.1'
done
echo "P11_DONE $(date -Is)"
