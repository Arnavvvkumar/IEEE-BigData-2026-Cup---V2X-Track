#!/bin/bash
# P10 (laptop, CPU): add the fine-grid VoxelNeXt models as TRUCK voters (they were only used for pedestrians / vehicles).
# On the test clip they find the large fixed vehicles with higher scores than several SECOND models. Base = v6 spec.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
PED="vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian"
T="canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck"
for s in dev150 dev350; do
  echo "=== $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s v6_norule vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle $PED $T norule
  "$P" rule_retune.py $s v6_norule 2 'a 0.5 base 0.1'
  # shellcheck disable=SC2086
  "$P" eval150.py $s v6t2_norule vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle,truck vnx010_pv_ft_ep5:vehicle,truck vnx010_pv_ft2_ep5:vehicle $PED $T norule
  "$P" rule_retune.py $s v6t2_norule 2 'a 0.5 base 0.1'
  # shellcheck disable=SC2086
  "$P" eval150.py $s v6t4_norule vnx010_all_s1234_ep10:pedestrian,vehicle,truck vnx010_twin_s1_ep12:pedestrian,vehicle,truck vnx010_pv_ft_ep5:vehicle,truck vnx010_pv_ft2_ep5:vehicle,truck $PED $T norule
  "$P" rule_retune.py $s v6t4_norule 2 'a 0.5 base 0.1'
done
echo "P10_DONE $(date -Is)"
