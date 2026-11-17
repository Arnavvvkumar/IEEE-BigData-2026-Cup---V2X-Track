#!/bin/bash
# Parallel GPU lane B: SECOND on the full canonical set, then VoxelNeXt 0.2 m on it.
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data/claude_canon_all_v1"
P="$HOME/projects/bdc2026-track-b/.env/bin/python"
T="$HOME/projects/detector-program/OpenPCDet/tools/cfgs/custom_models"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
bash "$here/run_arm_par.sh" canon_all_s1234 "$here/second_canon_all_v1.yaml" "$D" 12 "$G"
[ -f vnx020_all.yaml ] || "$P" make_cfg.py "$T/voxelnext_v1.yaml" "$D" vnx020_all.yaml score=0.05
BS=4 bash "$here/run_arm_par.sh" vnx020_all_s1234 "$here/vnx020_all.yaml" "$D" 12 "$G"
echo LANEB_DONE
