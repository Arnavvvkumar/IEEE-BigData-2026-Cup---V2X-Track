#!/bin/bash
# Lane M (P2): SECOND on canon + twin + small-truck twin (pickup/van/box-truck sized synthetic trucks), mixed precision.
# Same recipe as second_twin_s2 so the only change is the training data.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
export WORKERS=3
bash "$here/run_arm_par_ns.sh" second_twin_st_s1 "$here/second_canon_all_v1.yaml" "$D/claude_canon_all_twin_st_v1" 12 "$G" --use_amp
echo LANEM_DONE
