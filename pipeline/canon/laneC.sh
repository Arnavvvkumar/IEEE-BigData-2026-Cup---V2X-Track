#!/bin/bash
# Parallel GPU lane C: fine-grid (0.1 m) VoxelNeXt on the full canonical set (pedestrians), then PV-RCNN.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
bx="${BDC_ROOT}/experiments/claude_carla_overnight/boxsize"
D="${BDC_DATA}/claude_canon_all_v1"
P="${BDC_PYTHON}"
T="${BDC_PROGRAM}/OpenPCDet/tools/cfgs/custom_models"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
[ -f vnx010_all.yaml ] || "$P" make_cfg.py "$T/voxelnext_xy010_v1.yaml" "$D" vnx010_all.yaml score=0.05 train_voxels=60000 test_voxels=80000
BS=2 bash "$here/run_arm_par.sh" vnx010_all_s1234 "$here/vnx010_all.yaml" "$D" 12 "$G"
[ -f pvrcnn_all.yaml ] || "$P" make_cfg.py "$bx/pvrcnn_relabel_b1_v1.yaml" "$D" pvrcnn_all.yaml train_voxels=80000 test_voxels=80000
BS=1 bash "$here/run_arm_par.sh" pvrcnn_all_s1234 "$here/pvrcnn_all.yaml" "$D" 8 "$G"
echo LANEC_DONE
