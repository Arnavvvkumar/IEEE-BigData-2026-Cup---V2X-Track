#!/bin/bash
# P38: pedestrian fused-box GEOMETRY on the v16 base (v15 list + averaged pedestrian heading). Pre-registered:
# weights score^2, score^0.5; top-3 / top-6 members only. Fused scores unchanged. Pedestrian AP (hedged), dev150 + dev350 + halves.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
B="$B $(x3 pvrcnn_ped_P1_ep5) $(x3 pvrcnn_ped_P2_ep5) $(x3 pvrcnn_ped_P3_ep5) $(x3 pvrcnn_ped_P1_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5) pvrcnn_ped_P4_ep5:pedestrian"
cd "$base/coop"
ped() { env $3 "$P" eval150.py $1 $2 $B norule > /dev/null; "$P" verify_hedge_fn.py $1 $2 | tail -n 1 | sed "s/^/$1 $2 PED: /"; }
D="FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle"
echo "P38_START $(date -Is)"
for s in dev150 dev350; do
  ped $s pg_wpow2_norule "$D FUSE_PED_WPOW=2"
  ped $s pg_wpow05_norule "$D FUSE_PED_WPOW=0.5"
  ped $s pg_top3_norule "$D FUSE_PED_TOPK=3"
  ped $s pg_top6_norule "$D FUSE_PED_TOPK=6"
done
echo "P38_DONE $(date -Is)"
