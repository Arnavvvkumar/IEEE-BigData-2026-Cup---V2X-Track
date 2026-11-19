#!/bin/bash
# Parallel GPU lane D: second (random-seed) run of the current best recipe (SECOND, ground-canonical relabel mix).
# Purpose: seed-variance check on the 47% result and a second ensemble member.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
export WORKERS=2
bash "$here/run_arm_par_ns.sh" canon_relabel_rs2 "$here/second_canon_v1.yaml" "$D/claude_canon_relabel_mix6000_v1" 12 "$G"
echo LANED_DONE
