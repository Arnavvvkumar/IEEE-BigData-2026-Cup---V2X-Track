#!/bin/bash
# P41: only the dev150 survivors go to dev350 (full precision). dev150 survivors: robust vertical 50/50 (+0.18 all halves);
# score^0.5 weights (tie at 1 decimal -> checked at full precision first); top-6 (not yet run on dev150).
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
pkill -f "eval150.py dev" 2>/dev/null; sleep 2
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
B="$B $(x3 pvrcnn_ped_P1_ep5) $(x3 pvrcnn_ped_P2_ep5) $(x3 pvrcnn_ped_P3_ep5) $(x3 pvrcnn_ped_P1_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5) pvrcnn_ped_P4_ep5:pedestrian"
PB=""; for s in $B; do t=${s%%:*}; c=${s##*:}; [[ ",$c," == *",pedestrian,"* ]] && PB="$PB $t:pedestrian" || PB="$PB $t:none"; done
cd "$base/coop"
D="FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle"
run() { env $D $3 "$P" eval150.py $1 $2 $PB norule > /dev/null; "$P" ped_ap2.py $1 $2; }
echo "P41_START $(date -Is)"
"$P" ped_ap2.py dev150 pg_wpow05_norule
run dev150 px_top6_norule "FUSE_PED_TOPK=6" &
run dev350 px_base_norule "" &
run dev350 px_zmed50_norule "FUSE_PED_ZMED=0.5" &
run dev350 px_wpow05_norule "FUSE_PED_WPOW=0.5" &
wait
echo "P41_DONE $(date -Is)"
