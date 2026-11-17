#!/bin/bash
# GPU queue 4: takes over from queue 3 once both SECOND v3 arms are scored. Pedestrian-oriented fine grid first.
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
bx="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/boxsize"
P="$HOME/projects/bdc2026-track-b/.env/bin/python"
T="$HOME/projects/detector-program/OpenPCDet/tools/cfgs/custom_models"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
while ! grep -q 'canon_all_s1234 ep10 median ground -2.8' results/summary.txt 2>/dev/null; do sleep 20; done
pkill -f 'queue3\.sh'; sleep 2; pkill -f 'extra_tag canon_all_vnx_s1234'; sleep 5
# data set: the better of the two SECOND v3 arms by selection-half mAP (DSET file may be written by hand to override)
[ -f DSET ] || "$P" pick_set.py > DSET
D="$HOME/projects/bdc2026-track-b/data/$(cat DSET)"
n=$(cat DSET)
[ -f "vnx010_$n.yaml" ] || "$P" make_cfg.py "$T/voxelnext_xy010_v1.yaml" "$D" "vnx010_$n.yaml" score=0.05 train_voxels=60000 test_voxels=80000
BS=2 bash "$here/run_arm.sh" vnx010_s1234 "$here/vnx010_$n.yaml" "$D" 12 "$G"
[ -f "vnx020_$n.yaml" ] || "$P" make_cfg.py "$T/voxelnext_v1.yaml" "$D" "vnx020_$n.yaml" score=0.05
BS=4 bash "$here/run_arm.sh" vnx020_s1234 "$here/vnx020_$n.yaml" "$D" 12 "$G"
[ -f "pvrcnn_$n.yaml" ] || "$P" make_cfg.py "$bx/pvrcnn_relabel_b1_v1.yaml" "$D" "pvrcnn_$n.yaml" train_voxels=80000 test_voxels=80000
BS=1 bash "$here/run_arm.sh" pvrcnn_s1234 "$here/pvrcnn_$n.yaml" "$D" 8 "$G"
echo QUEUE4_DONE
