#!/bin/bash
# P20 (laptop): score T1's epoch-2 checkpoint (partly trained PV-RCNN truck-only) as a truck voter x1 and x3 on top of v9 bundle B. After the cancel script has copied it.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
CK="${BDC_PROGRAM}/OpenPCDet/output${BDC_ROOT}/experiments/claude_carla_overnight/canon/pvrcnn_all/pvrcnn_truck_T1/ckpt/checkpoint_epoch_2.pth"
while pgrep -f cancel_T1_after_ep2.sh > /dev/null; do sleep 30; done
[ -s "$CK" ] || { echo "NO_T1_CKPT"; exit 1; }
BATCH=1 bash "$base/final/run_infer_one.sh" pvrcnn_truck_T1 pvrcnn_truck.yaml claude_canon_all_twin_ll_v1 2
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V9="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck"
for s in dev150 dev350; do
  for w in 1 3; do
    x=$(for i in $(seq $w); do printf '%s ' pvrcnn_truck_T1_ep2:truck; done)
    # shellcheck disable=SC2086
    "$P" eval150.py $s t1e2w${w}_norule $V9 $x norule > /dev/null
    "$P" rule_retune.py $s t1e2w${w}_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s + T1 epoch 2 truck voter x$w: /"
  done
done
echo P20_DONE
