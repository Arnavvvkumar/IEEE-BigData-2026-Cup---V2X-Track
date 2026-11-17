#!/bin/bash
# P25: re-score V1 (some P24 runs ran out of memory with 8 in parallel). Waits for P24, then 2 at a time.
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
V10="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian"
export FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle
while pgrep -f "^bash p24_parallel.sh" > /dev/null; do sleep 10; done
echo "P25_START $(date -Is)"
cd "$base/coop"
for s in dev150 dev350; do
  for w in 1 3; do
    (x=$(for i in $(seq $w); do printf '%s ' "pvrcnn_veh_V1_ep5:vehicle"; done)
     "$P" eval150.py $s pvrcnn_veh_V1w${w}_norule $V10 $x norule > /dev/null
     "$P" rule_retune.py $s pvrcnn_veh_V1w${w}_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s + V1 x$w: /") &
  done
  wait
done
echo "P25_DONE $(date -Is)"
