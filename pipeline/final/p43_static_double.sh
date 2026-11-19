#!/bin/bash
# P43: Codex idea 2 (double penalty for sparse pedestrians). Re-fuse with FUSE_KEEP_SF=1 (px_sf_norule keeps each box's static fraction),
# check it reproduces the v17 reference, then run the 2 pre-registered variants. usage: p43_static_double.sh <dev150|dev350>
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
S=$1
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
B="$B $(x3 pvrcnn_ped_P1_ep5) $(x3 pvrcnn_ped_P2_ep5) $(x3 pvrcnn_ped_P3_ep5) $(x3 pvrcnn_ped_P1_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5) pvrcnn_ped_P4_ep5:pedestrian"
PB=""; for s in $B; do t=${s%%:*}; c=${s##*:}; [[ ",$c," == *",pedestrian,"* ]] && PB="$PB $t:pedestrian" || PB="$PB $t:none"; done
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle FUSE_KEEP_SF=1
echo "P43_START $S $(date -Is)"
"$P" eval150.py $S px_sf_norule $PB norule > /dev/null 2>&1
"$P" ped_static_double.py $S px_sf_norule
echo "P43_DONE $S $(date -Is)"
