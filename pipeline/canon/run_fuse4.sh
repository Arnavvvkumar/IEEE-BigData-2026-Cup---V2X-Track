#!/bin/bash
# Twin-trained SECOND: dump at six heights, then class-wise fusion variants with and without it. $1 = VoxelNeXt epoch.
set -uo pipefail
e=${1:-10}
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
P="$HOME/projects/detector-program/env310/bin/python"
G='-3.15 -3.075 -3.0 -2.925 -2.85 -2.775'
cd "$here"
F="$here/fusion"
[ -f "$F/second_twin_s1_ep12_allpred.json" ] || bash dump_arm.sh second_twin_s1 "$here/second_canon_all_v1.yaml" "$D/claude_canon_all_twin_v1" 12 "$G"
A="$F/canon_all_s1234_ep12_allpred.json"; B="$F/canon_relabel_s1234_ep12_allpred.json"; R2="$F/canon_all_rs2_ep12_allpred.json"
V="$F/vnx010_all_s1234_ep${e}_allpred.json"; T="$F/second_twin_s1_ep12_allpred.json"
VT="vehicle,truck"
echo "== twin SECOND alone, six heights"; "$P" fuse.py "$F/fuse_twin_heights.json" "$T" | grep fused
echo "== cw_b + twin (4 SECOND)"; "$P" fuse.py "$F/fuse_cw_b4_ep${e}.json" "$V::pedestrian,vehicle" "$A::$VT" "$R2::$VT" "$B::$VT" "$T::$VT" | grep fused
echo "== cw_b, twin replaces rs2"; "$P" fuse.py "$F/fuse_cw_bT2_ep${e}.json" "$V::pedestrian,vehicle" "$A::$VT" "$T::$VT" "$B::$VT" | grep fused
echo "== cw_b, twin replaces relabel"; "$P" fuse.py "$F/fuse_cw_bTr_ep${e}.json" "$V::pedestrian,vehicle" "$A::$VT" "$R2::$VT" "$T::$VT" | grep fused
echo "== ped from VNX, veh+truck twin + all_s1234"; "$P" fuse.py "$F/fuse_cw_bTA_ep${e}.json" "$V::pedestrian,vehicle" "$A::$VT" "$T::$VT" | grep fused
echo FUSE4_DONE
