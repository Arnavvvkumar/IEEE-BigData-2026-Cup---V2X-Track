#!/bin/bash
# P16: joint check of the v9 bundles with the FINAL PV-RCNN (job 1, epoch 6) replacing epoch 2:
#   A = v7 + pv6 pedestrian,vehicle      B = v7 + pv6 pedestrian,vehicle,truck
# Each scored with pedestrian hedging, the v5 truck rule, and also with the truck boost/hedge (calib.truck_boost_hedge).
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V7="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle"
for s in dev150 dev350; do
  for b in "A pvrcnn_st_osc_ep6:pedestrian,vehicle" "B pvrcnn_st_osc_ep6:pedestrian,vehicle,truck"; do
    n=${b%% *}; x=${b#* }
    # shellcheck disable=SC2086
    "$P" eval150.py $s v9$n\_norule $V7 $x norule > /dev/null
    "$P" verify_hedge_fn.py $s v9$n\_norule | tail -n 1
    "$P" rule_retune.py $s v9$n\_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s bundle $n: /"
    "$P" truck_variants.py $s v9$n\_norule 2 2>/dev/null | grep -E '^(base|long\+dense|hedge)' | sed "s/^/$s bundle $n trucks: /"
  done
done
echo P16_DONE
