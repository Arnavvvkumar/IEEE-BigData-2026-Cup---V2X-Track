#!/bin/bash
# One confidence vote per pass per cluster (Codex's vehicle finding), now tried for TRUCKS and vehicle+truck, on the v9 ensemble.
# Control = no ONE_VOTE (must reproduce v9: dev150 genuine 75.4 / dev350 74.7 with the C ensemble -> here the B ensemble + job 2 = v9).
cd "${BDC_ROOT}/experiments/claude_carla_overnight/coop"
P="${BDC_PYTHON}"
export FUSE_YAW_CIRC=vehicle
V9="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle"
for s in dev150 dev350; do
  for ov in none truck vehicle vehicle,truck; do
    o=$ov; [ $o = none ] && o=''
    # shellcheck disable=SC2086
    FUSE_ONE_VOTE=$o "$P" eval150.py $s ov_norule $V9 norule > /dev/null
    "$P" rule_retune.py $s ov_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s one-vote [$ov]: /"
  done
done
echo ONEVOTE_DONE
