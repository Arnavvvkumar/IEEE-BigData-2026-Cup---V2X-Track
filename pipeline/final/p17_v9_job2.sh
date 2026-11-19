#!/bin/bash
# P17: does job 2 (PV-RCNN seed 2, epoch 4) still help on top of v9 (bundle B: v7 + job 1 epoch 6 for all classes)?
# Tried as pedestrian+vehicle voter (C) and pedestrian+vehicle+truck voter (D). Truck hedging is neutral, so scored without it.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V9="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck"
for s in dev150 dev350; do
  for b in "C pvrcnn_st_osc_s2_ep4:pedestrian,vehicle" "D pvrcnn_st_osc_s2_ep4:pedestrian,vehicle,truck"; do
    n=${b%% *}; x=${b#* }
    # shellcheck disable=SC2086
    "$P" eval150.py $s v9$n\_norule $V9 $x norule > /dev/null
    "$P" verify_hedge_fn.py $s v9$n\_norule | tail -n 1
    "$P" rule_retune.py $s v9$n\_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s bundle $n: /"
  done
done
echo P17_DONE
