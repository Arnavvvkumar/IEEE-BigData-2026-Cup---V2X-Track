#!/bin/bash
# Lane O (vehicles + trucks): SECOND on the full merged set canon + twin + small-truck twin + size-varied twin (23,800 frames),
# 14 epochs, mixed precision. A second small-truck-aware SECOND member (different seed, more twin data) for the vehicle/truck ensemble.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
export WORKERS=3
bash "$here/run_arm_par_ns.sh" second_twin_pv_s1 "$here/second_canon_all_v1.yaml" "$D/claude_canon_all_twin_pv_v1" 14 "$G" --use_amp
echo LANEO_DONE
