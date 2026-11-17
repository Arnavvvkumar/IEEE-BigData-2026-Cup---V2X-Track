#!/bin/bash
# Lane K: second seed of SECOND on the merged twin set, mixed precision (twin SECOND gave the vehicle gain in fusion).
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
sleep 400
export WORKERS=3
bash "$here/run_arm_par_ns.sh" second_twin_s2 "$here/second_canon_all_v1.yaml" "$D/claude_canon_all_twin_v1" 12 "$G" --use_amp
echo LANEK_DONE
