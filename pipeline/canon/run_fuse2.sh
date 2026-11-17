#!/bin/bash
# Early look at the fine-grid VoxelNeXt (epoch given as $1) for pedestrians: dump at six heights, fuse alone and with the SECOND models.
set -uo pipefail
e=${1:-10}
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
P="$HOME/projects/detector-program/env310/bin/python"
G='-3.15 -3.075 -3.0 -2.925 -2.85 -2.775'
cd "$here"
bash dump_arm.sh vnx010_all_s1234 "$here/vnx010_all.yaml" "$D/claude_canon_all_v1" "$e" "$G"
F="$here/fusion"
echo "== heights, vnx010 ep$e"; "$P" fuse.py "$F/fuse_vnx010_ep${e}_heights.json" "$F/vnx010_all_s1234_ep${e}_allpred.json" | grep fused
echo "== vnx010 + 2 SECOND, all heights"; "$P" fuse.py "$F/fuse_3model_ep${e}.json" "$F/vnx010_all_s1234_ep${e}_allpred.json" "$F/canon_all_s1234_ep12_allpred.json" "$F/canon_relabel_s1234_ep12_allpred.json" | grep fused
echo FUSE2_DONE
