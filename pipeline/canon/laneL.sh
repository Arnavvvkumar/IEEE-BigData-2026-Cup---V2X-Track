#!/bin/bash
# Lane L: second seed of the fine-grid VoxelNeXt on the merged twin set (the first seed is the strongest single model), AMP.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
sleep 240
export WORKERS=3
BS=2 bash "$here/run_arm_par_ns.sh" vnx010_twin_s2 "$here/vnx010_all.yaml" "$D/claude_canon_all_twin_v1" 12 "$G" --use_amp
echo LANEL_DONE
