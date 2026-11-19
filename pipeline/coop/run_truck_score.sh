#!/bin/bash
# Truck fusion-score aggregates on the v9 bundle-B ensemble (both dev sets). Scales chosen so native truck scores stay comparable to
# vehicle-copy scores: max x0.12, mean x0.6, geo x0.25 (sum/passes is about 0.1x the max for an object every truck model finds).
cd "${BDC_ROOT}/experiments/claude_carla_overnight/coop"
P="${BDC_PYTHON}"
export FUSE_YAW_CIRC=vehicle
V9="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck"
for s in dev150 dev350; do
  for m in max:0.12 max:0.2 mean:0.6 geo:0.25; do
    # shellcheck disable=SC2086
    FUSE_TRUCK_SCORE=$m "$P" eval150.py $s ts_norule $V9 norule > /dev/null
    "$P" rule_retune.py $s ts_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s truck score $m: /"
  done
done
echo TRUCK_SCORE_DONE
