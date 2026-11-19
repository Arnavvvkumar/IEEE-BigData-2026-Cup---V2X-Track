#!/bin/bash
# GPU queue 3 (replaces queue 2): SECOND on the two CARLA-v3 sets first (fast, most informative), then VoxelNeXt and PV-RCNN.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
bx="${BDC_ROOT}/experiments/claude_carla_overnight/boxsize"
D="${BDC_DATA}"
P="${BDC_PYTHON}"
T="${BDC_PROGRAM}/OpenPCDet/tools/cfgs/custom_models"
G='-3.2 -3.0 -2.8'
cd "$here"
while ! grep -q SETS_DONE build_v3_sets.log 2>/dev/null; do sleep 30; done
bash "$here/run_arm.sh" canon_carlaonly_s1234 "$here/second_canon_carlaonly_v1.yaml" "$D/claude_canon_carlaonly_v1" 12 "$G"
bash "$here/run_arm.sh" canon_all_s1234 "$here/second_canon_all_v1.yaml" "$D/claude_canon_all_v1" 12 "$G"
[ -f voxelnext_canon_all_v1.yaml ] || "$P" make_cfg.py "$T/voxelnext_v1.yaml" "$D/claude_canon_all_v1" voxelnext_canon_all_v1.yaml score=0.05
BS=4 bash "$here/run_arm.sh" canon_all_vnx_s1234 "$here/voxelnext_canon_all_v1.yaml" "$D/claude_canon_all_v1" 16 "$G"
[ -f pvrcnn_canon_all_v1.yaml ] || "$P" make_cfg.py "$bx/pvrcnn_relabel_b1_v1.yaml" "$D/claude_canon_all_v1" pvrcnn_canon_all_v1.yaml train_voxels=80000 test_voxels=80000
BS=1 bash "$here/run_arm.sh" canon_all_pvrcnn_s1234 "$here/pvrcnn_canon_all_v1.yaml" "$D/claude_canon_all_v1" 8 "$G"
echo QUEUE3_DONE
