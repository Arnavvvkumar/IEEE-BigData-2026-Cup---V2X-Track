#!/bin/bash
# P45: Codex idea 2 - bimodal pedestrian clusters. Fast ped-only path, v17 recipe, 3 decimals. REF = loc_ref (46.554 dev150 / 45.929 dev350).
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
S=${1:-dev150}
B="vnx010_all_s1234_ep10 vnx010_twin_s1_ep12 vnx010_all_rs2_amp_ep12 vnx010_twin_s1_tta_ep12 vnx010_all_s1234_tta_ep10 pvrcnn_st_osc_ep6 pvrcnn_st_osc_s2_ep4"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
BASEPED=""; for t in $B; do BASEPED="$BASEPED $t:pedestrian"; done
for t in pvrcnn_ped_P1_ep5 pvrcnn_ped_P2_ep5 pvrcnn_ped_P3_ep5 pvrcnn_ped_P1_tta_ep5 pvrcnn_ped_P3_tta_ep5; do BASEPED="$BASEPED $(x3 $t)"; done
BASEPED="$BASEPED pvrcnn_ped_P4_ep5:pedestrian"
NONPED="vnx010_pv_ft_ep5 vnx010_pv_ft2_ep5 canon_all_s1234_ep12 second_twin_s1_ep12 canon_relabel_s1234_ep12 second_twin_s2_ep12 second_twin_st_s1_ep12 second_twin_pv_s1_ep14 vnx010_pv_ft4_ep4"
NP=""; for t in $NONPED; do NP="$NP $t:none"; done
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle
one() {   # $1 name, $2 mode, $3 sep
  FUSE_PED_MODE=$2 FUSE_PED_MODE_D=$3 FUSE_AUDIT="$base/coop/$S/audit_$1.json" "$P" eval150.py $S $1 $NP $BASEPED norule > /dev/null 2>&1 || echo "EVAL_FAIL $1"
  "$P" ped_ap3.py $S $1; cat "$base/coop/$S/audit_$1.json"; echo
}
echo "P45_START $S $(date -Is)"
one bm_M1_d35 1 0.35 > /tmp/p45_${S}_1.txt 2>&1 &
one bm_M2_d35 2 0.35 > /tmp/p45_${S}_2.txt 2>&1 &
one bm_M1_d60 1 0.60 > /tmp/p45_${S}_3.txt 2>&1 &
wait
cat /tmp/p45_${S}_1.txt /tmp/p45_${S}_2.txt /tmp/p45_${S}_3.txt
echo "P45_DONE $S $(date -Is)"
