#!/bin/bash
# P2: score the small-truck SECOND (second_twin_st_s1 ep12) in the ensemble, with averaged vehicle heading + truck de-duplication.
# One eval at a time (WSL has 11 GB and trainings are running).
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
echo "P2_START $(date -Is)"
bash "$base/final/run_infer_one.sh" second_twin_st_s1 second_canon_all_v1.yaml claude_canon_all_twin_st_v1 12
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V3="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian vnx010_twin_s2_ep12:pedestrian"
A="canon_all_s1234_ep12:vehicle,truck"; B="canon_relabel_s1234_ep12:vehicle,truck"
T1="second_twin_s1_ep12:vehicle,truck"; T2="second_twin_s2_ep12:vehicle,truck"; ST="second_twin_st_s1_ep12:vehicle,truck"
for s in dev150 dev350; do
  echo "=== eval $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s st_add     $V3 $A $T1 $B $T2 $ST dedupe
  # shellcheck disable=SC2086
  "$P" eval150.py $s st_forB    $V3 $A $T1 $T2 $ST dedupe
  # shellcheck disable=SC2086
  "$P" eval150.py $s st_forA    $V3 $T1 $B $T2 $ST dedupe
  # shellcheck disable=SC2086
  "$P" eval150.py $s st_truckonly $V3 $A $T1 $B $T2 second_twin_st_s1_ep12:truck dedupe
done
echo "P2_DONE $(date -Is)"
