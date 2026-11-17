#!/bin/bash
# Lane G: second (random-seed) fine-grid VoxelNeXt with mixed precision. The first one tripled pedestrian AP over SECOND
# (27-29 vs 12 at epoch 10), so a second member should steady the pedestrian class in the fusion.
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
export WORKERS=3
BS=2 bash "$here/run_arm_par_ns.sh" vnx010_all_rs2_amp "$here/vnx010_all.yaml" "$D/claude_canon_all_v1" 12 "$G" --use_amp
echo LANEG_DONE
