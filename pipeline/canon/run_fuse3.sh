#!/bin/bash
# Seed-2 full-set SECOND: dump at six heights (its scoring crashed on a busy GPU), then class-wise fusion:
# pedestrians from the fine-grid VoxelNeXt, trucks from SECOND, vehicles from both. $1 = VoxelNeXt epoch.
set -uo pipefail
e=${1:-10}
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
P="$HOME/projects/detector-program/env310/bin/python"
G='-3.15 -3.075 -3.0 -2.925 -2.85 -2.775'
cd "$here"
F="$here/fusion"
[ -f "$F/canon_all_rs2_ep12_allpred.json" ] || bash dump_arm.sh canon_all_rs2 "$here/second_canon_all_v1.yaml" "$D/claude_canon_all_v1" 12 "$G"
A="$F/canon_all_s1234_ep12_allpred.json"; B="$F/canon_relabel_s1234_ep12_allpred.json"; R2="$F/canon_all_rs2_ep12_allpred.json"; V="$F/vnx010_all_s1234_ep${e}_allpred.json"
echo "== SECOND x3 (all, rs2, relabel)"; "$P" fuse.py "$F/fuse_second3.json" "$A" "$R2" "$B" | grep fused
echo "== classwise: ped from VNX, veh+truck from SECOND x3"; "$P" fuse.py "$F/fuse_cw_a_ep${e}.json" "$V::pedestrian" "$A::vehicle,truck" "$R2::vehicle,truck" "$B::vehicle,truck" | grep fused
echo "== classwise: ped from VNX, veh from all, truck from SECOND x3"; "$P" fuse.py "$F/fuse_cw_b_ep${e}.json" "$V::pedestrian,vehicle" "$A::vehicle,truck" "$R2::vehicle,truck" "$B::vehicle,truck" | grep fused
echo "== classwise: ped from VNX + SECOND, veh from all, truck from SECOND"; "$P" fuse.py "$F/fuse_cw_c_ep${e}.json" "$V::pedestrian,vehicle" "$A" "$R2" "$B" | grep fused
echo FUSE3_DONE
