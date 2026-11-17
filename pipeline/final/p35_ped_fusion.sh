#!/bin/bash
# P35: pedestrian fusion settings never tuned for the current ensemble (v15s list). Pedestrian AP only (with hedging), dev150 + dev350 + halves.
# Pre-registered: one-vote for pedestrian, circular-mean yaw for pedestrian, cluster radius 0.3 / 0.5 (default 0.4), static down 0.6 / 1.0 (default 0.8).
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
B="$B $(x3 pvrcnn_ped_P1_ep5) $(x3 pvrcnn_ped_P2_ep5) $(x3 pvrcnn_ped_P3_ep5) $(x3 pvrcnn_ped_P1_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5) pvrcnn_ped_P4_ep5:pedestrian"
cd "$base/coop"
ped() { env $3 "$P" eval150.py $1 $2 $B norule > /dev/null; "$P" verify_hedge_fn.py $1 $2 | tail -n 1 | sed "s/^/$1 $2 PED: /"; }
echo "P35_START $(date -Is)"
D="FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle"
for s in dev150 dev350; do
  ped $s pf_base_norule "$D"
  ped $s pf_onevote_norule "FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle,pedestrian"
  ped $s pf_yawcirc_norule "FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle"
  ped $s pf_r03_norule "$D FUSE_PED_RADIUS=0.3"
  ped $s pf_r05_norule "$D FUSE_PED_RADIUS=0.5"
  ped $s pf_down06_norule "$D FUSE_STATIC_DOWN=0.6"
  ped $s pf_down10_norule "$D FUSE_STATIC_DOWN=1.0"
done
echo "P35_DONE $(date -Is)"
