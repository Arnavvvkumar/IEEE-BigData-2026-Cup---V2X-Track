#!/bin/bash
# Early look: VR2 on dev150 only (pedestrian AP, v16 base = v15 list + averaged pedestrian heading). Full gate stays in P36.
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
B="$B $(x3 pvrcnn_ped_P1_ep5) $(x3 pvrcnn_ped_P2_ep5) $(x3 pvrcnn_ped_P3_ep5) $(x3 pvrcnn_ped_P1_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5) pvrcnn_ped_P4_ep5:pedestrian"
export FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle
cd "$base/coop"
ped() { "$P" eval150.py $1 $2 $3 norule > /dev/null; "$P" verify_hedge_fn.py $1 $2 | tail -n 1 | sed "s/^/$1 $2 PED: /"; }
t=vrcnn_ped_VR2_ep5
ped dev150 e_VR2_x1_norule "$B $t:pedestrian"
ped dev150 e_VR2_x3_norule "$B $(x3 $t)"
echo "standalone VR2:"; ped dev150 e_VR2_alone_norule "$t:pedestrian"
echo P37_DONE
