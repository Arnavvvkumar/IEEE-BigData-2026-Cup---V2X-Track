#!/bin/bash
# Lane N (pedestrians): fine-grid VoxelNeXt, initialised from our own synthetic-trained vnx010_twin_s1 (epoch 12), then trained
# 5 epochs on canon + twin + small-truck twin + SIZE-VARIED pedestrian twin, so it learns to read pedestrian box size from the points.
# Synthetic-only: the initial weights were trained on synthetic data only, and so is this run.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
repo="${BDC_PROGRAM}/OpenPCDet"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
init=$(find "$repo/output" -path "*/vnx010_twin_s1/ckpt/checkpoint_epoch_12.pth" -print -quit)
echo "init $init"
[ -f "$init" ] || { echo "NO_INIT_CKPT"; exit 1; }
export WORKERS=3
BS=2 bash "$here/run_arm_par_ns.sh" vnx010_pv_ft "$here/vnx010_all.yaml" "$D/claude_canon_all_twin_pv_v1" 5 "$G" --use_amp --pretrained_model "$init"
echo LANEN_DONE
