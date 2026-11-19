#!/bin/bash
# GPU queue 2 (after queue 1): VoxelNeXt, then PV-RCNN, both on the ground-canonical relabelled 6k mix.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
bx="${BDC_ROOT}/experiments/claude_carla_overnight/boxsize"
D="${BDC_DATA}/claude_canon_relabel_mix6000_v1"
P="${BDC_PYTHON}"
T="${BDC_PROGRAM}/OpenPCDet/tools/cfgs/custom_models"
cd "$here"
[ -f voxelnext_canon_v1.yaml ] || "$P" make_cfg.py "$T/voxelnext_v1.yaml" "$D" voxelnext_canon_v1.yaml score=0.05
[ -f pvrcnn_canon_v1.yaml ] || "$P" make_cfg.py "$bx/pvrcnn_relabel_b1_v1.yaml" "$D" pvrcnn_canon_v1.yaml train_voxels=80000 test_voxels=80000
sleep 60
BS=4 bash "$here/run_arm.sh" canon_vnx_s1234 "$here/voxelnext_canon_v1.yaml" "$D" 20 '-3.2 -3.0 -2.8'
BS=1 bash "$here/run_arm.sh" canon_pvrcnn_s1234 "$here/pvrcnn_canon_v1.yaml" "$D" 12 '-3.2 -3.0 -2.8'
