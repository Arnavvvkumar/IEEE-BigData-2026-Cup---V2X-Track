#!/bin/bash
# Parallel GPU lane E: second (random-seed) run of the new best recipe (SECOND, full canonical set) for an ensemble member.
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
export WORKERS=2
bash "$here/run_arm_par_ns.sh" canon_all_rs2 "$here/second_canon_all_v1.yaml" "$D/claude_canon_all_v1" 12 "$G"
echo LANEE_DONE
