#!/bin/bash
# Pedestrian test-time shift/rotation passes added to the best ensemble (e_addVT) on the 150 validation frames.
cd "${BDC_ROOT}/experiments/claude_carla_overnight/coop"
P="${BDC_PYTHON}"
V=vnx010_all_s1234_ep10; V2=vnx010_all_rs2_amp_ep12; VT=vnx010_twin_s1_ep12
A=canon_all_s1234_ep12; T=second_twin_s1_ep12; B=canon_relabel_s1234_ep12
TV=vnx010_all_s1234_tta_ep10; TT=vnx010_twin_s1_tta_ep12
BASE="$V:pedestrian,vehicle $V2:pedestrian $VT:pedestrian,vehicle $A:vehicle,truck $T:vehicle,truck $B:vehicle,truck"
"$P" eval150.py dev150 i_ttaTwinPed  $BASE $TT:pedestrian
"$P" eval150.py dev150 j_ttaBothPed  $BASE $TT:pedestrian $TV:pedestrian
"$P" eval150.py dev150 k_ttaBothPV   $BASE $TT:pedestrian,vehicle $TV:pedestrian,vehicle
echo EVAL_B_DONE
