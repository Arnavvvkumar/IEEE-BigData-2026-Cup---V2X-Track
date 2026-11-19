#!/bin/bash
# Parallel GPU lane C (v2): fine-grid VoxelNeXt (pedestrians) -> SECOND CARLA-only (resume) -> PV-RCNN. Lane B runs alongside.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
bx="${BDC_ROOT}/experiments/claude_carla_overnight/boxsize"
DD="${BDC_DATA}"
D="$DD/claude_canon_all_v1"
P="${BDC_PYTHON}"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
export WORKERS=3
BS=2 bash "$here/run_arm_par.sh" vnx010_all_s1234 "$here/vnx010_all.yaml" "$D" 12 "$G"
bash "$here/run_arm_par.sh" canon_carlaonly_s1234 "$here/second_canon_carlaonly_v1.yaml" "$DD/claude_canon_carlaonly_v1" 12 "$G"
[ -f pvrcnn_all.yaml ] || "$P" make_cfg.py "$bx/pvrcnn_relabel_b1_v1.yaml" "$D" pvrcnn_all.yaml train_voxels=80000 test_voxels=80000
BS=1 bash "$here/run_arm_par.sh" pvrcnn_all_s1234 "$here/pvrcnn_all.yaml" "$D" 8 "$G"
echo LANEC2_DONE
