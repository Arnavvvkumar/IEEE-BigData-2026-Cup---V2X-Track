#!/bin/bash
# P23 (laptop; OSC read-only): speed-up of P22's V1 half. Fetch V1 now (network) while P2 infers, infer V1 right after P2's inference,
# score the 4 V1 settings in parallel, then stop P22 if it has started its own (duplicate) V1 half.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
tag=pvrcnn_veh_V1; cfg=pvrcnn_veh.yaml; last=5
V10="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian"
export FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle
echo "P23_START $(date -Is)"
bash "$base/osc/fetch_pv_ckpt.sh" "$last" "$tag" | tail -n 1
while pgrep -f "run_infer_one.sh pvrcnn_ped_P2" > /dev/null; do sleep 20; done
echo "P2 inference done $(date -Is)"
BATCH=1 bash "$base/final/run_infer_one.sh" "$tag" "$cfg" claude_canon_all_twin_st_v1 "$last" | tail -n 3
echo "V1 inference done $(date -Is)"
cd "$base/coop"
for s in dev150 dev350; do
  for w in 1 3; do
    (x=$(for i in $(seq $w); do printf '%s ' "${tag}_ep$last:vehicle"; done)
     "$P" eval150.py $s ${tag}w${w}_norule $V10 $x norule > /dev/null
     "$P" rule_retune.py $s ${tag}w${w}_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s + $tag x$w: /") &
  done
done
wait
echo "P23_DONE $(date -Is)"
