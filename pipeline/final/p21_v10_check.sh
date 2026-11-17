#!/bin/bash
# P21: (1) P1 (pedestrian-only PV-RCNN, epoch 5) vote weight 2 and 5; (2) JOINT check of the v10 candidate on both dev sets:
# v9 ensemble + P1 x3 (pedestrian) + FUSE_ONE_VOTE=vehicle. Pedestrian AP with hedging; vehicle/truck AP with the v5 truck rule.
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V9="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle"
P1=pvrcnn_ped_P1_ep5:pedestrian
rep() { for i in $(seq $1); do printf '%s ' $P1; done; }
for s in dev150 dev350; do
  for w in 2 5; do
    # shellcheck disable=SC2086
    FUSE_ONE_VOTE= "$P" eval150.py $s p1w${w}_norule $V9 $(rep $w) norule > /dev/null
    "$P" verify_hedge_fn.py $s p1w${w}_norule | tail -n 1 | sed "s/^/weight $w: /"
  done
done
for s in dev150 dev350; do
  # shellcheck disable=SC2086
  FUSE_ONE_VOTE=vehicle "$P" eval150.py $s v10_norule $V9 $(rep 3) norule > /dev/null
  "$P" verify_hedge_fn.py $s v10_norule | tail -n 1 | sed "s/^/v10 joint: /"
  "$P" rule_retune.py $s v10_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s v10 joint: /"
done
echo P21_DONE
