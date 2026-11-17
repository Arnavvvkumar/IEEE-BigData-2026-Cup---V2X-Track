#!/bin/bash
# P31: Codex truck audit - the SECOND small-truck-twin voter (second_twin_st_s1) natively finds 73/81 small trucks but votes once.
# Test its truck weight x2 / x3 on top of the v13 list (+P1 TTA x3). Truck/genuine-truck gate on dev150/dev350 + halves; other classes must not drop.
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P1_tta_ep5:pedestrian pvrcnn_ped_P1_tta_ep5:pedestrian pvrcnn_ped_P1_tta_ep5:pedestrian"
ST="second_twin_st_s1_ep12:truck"
export FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle
cd "$base/coop"
vt() { "$P" eval150.py $1 $2 $3 norule > /dev/null; "$P" rule_retune.py $1 $2 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$1 $2 VT: /"; }
echo "P31_START $(date -Is)"
for s in dev150 dev350; do
  vt $s st1_norule "$B" &
  vt $s st2_norule "$B $ST" &
  wait
  vt $s st3_norule "$B $ST $ST"
done
echo "P31_DONE $(date -Is)"
