#!/bin/bash
# Lane I: fine-grid VoxelNeXt on the merged set (canonical full set + site twin), mixed precision.
# Starts once the merged dataset exists and the VoxelNeXt 0.2 m training has finished (keeps three trainings on the GPU).
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
while [ ! -f "$D/claude_canon_all_twin_v1/manifest.json" ]; do sleep 20; done
while pgrep -f 'extra_tag vnx020_all_s1234' > /dev/null; do sleep 30; done
sleep 90
export WORKERS=3
BS=2 bash "$here/run_arm_par_ns.sh" vnx010_twin_s1 "$here/vnx010_all.yaml" "$D/claude_canon_all_twin_v1" 12 "$G" --use_amp
echo LANEI_DONE
