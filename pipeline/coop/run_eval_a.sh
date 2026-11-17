#!/bin/bash
# Ensemble comparisons on the 150 validation frames once the twin VoxelNeXt predictions exist.
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop"
P="$HOME/projects/detector-program/env310/bin/python"
while [ ! -f dev150/allpred/vnx010_twin_s1_ep12_allpred.json ]; do sleep 20; done
sleep 5
V=vnx010_all_s1234_ep10; V2=vnx010_all_rs2_amp_ep12; VT=vnx010_twin_s1_ep12
A=canon_all_s1234_ep12; T=second_twin_s1_ep12; B=canon_relabel_s1234_ep12
"$P" eval150.py dev150 d_rule    $V:pedestrian,vehicle $V2:pedestrian $A:vehicle,truck $T:vehicle,truck $B:vehicle,truck
"$P" eval150.py dev150 e_addVT   $V:pedestrian,vehicle $V2:pedestrian $VT:pedestrian,vehicle $A:vehicle,truck $T:vehicle,truck $B:vehicle,truck
"$P" eval150.py dev150 f_VTswapV $VT:pedestrian,vehicle $V2:pedestrian $V:pedestrian $A:vehicle,truck $T:vehicle,truck $B:vehicle,truck
"$P" eval150.py dev150 g_VTall   $VT:pedestrian,vehicle,truck $V:pedestrian,vehicle $V2:pedestrian $A:vehicle,truck $T:vehicle,truck $B:vehicle,truck
"$P" eval150.py dev150 h_VTveh   $VT:pedestrian,vehicle $V:pedestrian $V2:pedestrian $A:truck $T:vehicle,truck $B:truck
echo EVAL_A_DONE
