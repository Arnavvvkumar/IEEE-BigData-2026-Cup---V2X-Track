#!/bin/bash
# Lane J: third fine-grid VoxelNeXt seed on the canonical full set (old data), mixed precision. Pedestrian ensemble member.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
while pgrep -f 'detilt_eval.py' > /dev/null; do sleep 20; done
sleep 60
export WORKERS=3
BS=2 bash "$here/run_arm_par_ns.sh" vnx010_all_rs3_amp "$here/vnx010_all.yaml" "$D/claude_canon_all_v1" 12 "$G" --use_amp
echo LANEJ_DONE
