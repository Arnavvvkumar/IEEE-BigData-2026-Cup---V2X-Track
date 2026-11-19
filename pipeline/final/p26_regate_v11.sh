#!/bin/bash
# P26: re-gate P2 / V1 against the actual v11 voter list (v10 list WITHOUT the long-low voter). Waits for P25. 2 jobs at a time.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian"
P2="pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian"
V1="pvrcnn_veh_V1_ep5:vehicle"
export FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle
while pgrep -f "^bash p25_V1_rescore.sh" > /dev/null; do sleep 10; done
echo "P26_START $(date -Is)"
cd "$base/coop"
score() {  # set name passes
  "$P" eval150.py $1 $2 $3 norule > /dev/null
  "$P" verify_hedge_fn.py $1 $2 | tail -n 1 | sed "s/^/$1 $2 PED: /"
  "$P" rule_retune.py $1 $2 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$1 $2 VT: /"
}
for s in dev150 dev350; do
  score $s v11base_norule "$B" &
  score $s v11p2_norule "$B $P2" &
  wait
  score $s v11p2v1_norule "$B $P2 $V1" &
  wait
done
echo "P26_DONE $(date -Is)"
