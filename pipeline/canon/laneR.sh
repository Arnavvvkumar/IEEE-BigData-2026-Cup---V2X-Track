#!/bin/bash
# Lane R (vehicle/pedestrian voter): fourth fine-grid VoxelNeXt fine-tune on the full merged set, initialised from our own
# synthetic-trained vnx010_all_rs2_amp (epoch 12). Each such fine-tune has added a little as an ensemble voter.
# Synthetic-only: the initial weights were trained on synthetic data only, and so is this run.
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
repo="$HOME/projects/detector-program/OpenPCDet"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
init=$(find "$repo/output" -path "*/vnx010_all_rs2_amp/ckpt/checkpoint_epoch_12.pth" -print -quit)
echo "init $init"
[ -f "$init" ] || { echo "NO_INIT_CKPT"; exit 1; }
export WORKERS=3
BS=2 bash "$here/run_arm_par_ns.sh" vnx010_pv_ft4 "$here/vnx010_all.yaml" "$D/claude_canon_all_twin_pv_v1" 4 "$G" --use_amp --pretrained_model "$init"
echo LANER_DONE
