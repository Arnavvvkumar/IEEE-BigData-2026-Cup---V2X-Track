#!/bin/bash
# P13: score long-low truck seed 2 on top of v7 now (does not wait for the PV-RCNN checkpoint copy). One eval at a time.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
V7="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle"
export FUSE_YAW_CIRC=vehicle
echo "P13_START $(date -Is)"
bash "$base/final/run_infer_one.sh" second_twin_ll_s2 second_canon_all_v1.yaml claude_canon_all_twin_ll_v1 12
cd "$base/coop"
for s in dev150 dev350; do
  echo "=== ll seed 2 on $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s ll2_norule $V7 second_twin_ll_s2_ep12:truck norule > /dev/null
  "$P" rule_retune.py $s ll2_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed 's/^/+ ll seed 2 truck voter: /'
done
echo "P13_DONE $(date -Is)"
