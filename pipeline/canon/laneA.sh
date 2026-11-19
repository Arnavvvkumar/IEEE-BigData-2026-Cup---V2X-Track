#!/bin/bash
# Parallel GPU lane A: SECOND on CARLA only (resumes from its last saved epoch).
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
export WORKERS=2
bash "$here/run_arm_par.sh" canon_carlaonly_s1234 "$here/second_canon_carlaonly_v1.yaml" "$D/claude_canon_carlaonly_v1" 12 "$G"
echo LANEA_DONE
