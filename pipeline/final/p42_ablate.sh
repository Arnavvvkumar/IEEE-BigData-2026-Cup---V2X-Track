#!/bin/bash
# P42: leave-one-voter-group-out ablation of the PEDESTRIAN ensemble (never done: voters were added greedily). Fast pedestrian-only path
# (vehicle/truck passes kept as empty 'none' passes), v17 recipe (min-points rule + hedging), full precision.
# usage: p42_ablate.sh <dev150|dev350> <parallel> [group ...]    (no groups = all 13)
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
S=$1; PAR=$2; shift 2
declare -A G
G[vnx_all]="vnx010_all_s1234_ep10"
G[vnx_twin]="vnx010_twin_s1_ep12"
G[vnx_rs2]="vnx010_all_rs2_amp_ep12"
G[vnx_twin_tta]="vnx010_twin_s1_tta_ep12"
G[vnx_all_tta]="vnx010_all_s1234_tta_ep10"
G[pv_job1]="pvrcnn_st_osc_ep6"
G[pv_job2]="pvrcnn_st_osc_s2_ep4"
G[P1]="pvrcnn_ped_P1_ep5 pvrcnn_ped_P1_ep5 pvrcnn_ped_P1_ep5"
G[P2]="pvrcnn_ped_P2_ep5 pvrcnn_ped_P2_ep5 pvrcnn_ped_P2_ep5"
G[P3]="pvrcnn_ped_P3_ep5 pvrcnn_ped_P3_ep5 pvrcnn_ped_P3_ep5"
G[P1tta]="pvrcnn_ped_P1_tta_ep5 pvrcnn_ped_P1_tta_ep5 pvrcnn_ped_P1_tta_ep5"
G[P3tta]="pvrcnn_ped_P3_tta_ep5 pvrcnn_ped_P3_tta_ep5 pvrcnn_ped_P3_tta_ep5"
G[P4]="pvrcnn_ped_P4_ep5"
ORDER="vnx_all vnx_twin vnx_rs2 vnx_twin_tta vnx_all_tta pv_job1 pv_job2 P1 P2 P3 P1tta P3tta P4"
NONPED="vnx010_pv_ft_ep5 vnx010_pv_ft2_ep5 canon_all_s1234_ep12 second_twin_s1_ep12 canon_relabel_s1234_ep12 second_twin_s2_ep12 second_twin_st_s1_ep12 second_twin_pv_s1_ep14 vnx010_pv_ft4_ep4"
[ $# -gt 0 ] && ORDER="$*"
spec_without() {   # $1 = dropped group ('' = none) -> prints the pass spec list
  local out=""
  for t in $NONPED; do out="$out $t:none"; done
  for g in vnx_all vnx_twin vnx_rs2 vnx_twin_tta vnx_all_tta pv_job1 pv_job2 P1 P2 P3 P1tta P3tta P4; do
    for t in ${G[$g]}; do
      if [ "$g" = "$1" ]; then out="$out $t:none"; else out="$out $t:pedestrian"; fi
    done
  done
  echo $out
}
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle
one() {
  local g=$1 name="ab_${1}_norule"
  "$P" eval150.py $S $name $(spec_without $g) norule > /dev/null 2>&1
  "$P" ped_ap3.py $S $name
}
export -f one spec_without; export P S base NONPED
declare -p G > /tmp/_g.sh
echo "P42_START $S $(date -Is)"
printf '%s\n' $ORDER | xargs -P $PAR -I{} bash -c 'source /tmp/_g.sh; one {}'
echo "P42_DONE $S $(date -Is)"
