#!/bin/bash
# Lane P (pedestrians): second size-aware fine-grid VoxelNeXt. Same recipe as Lane N but initialised from our own
# synthetic-trained vnx010_all_s1234 (epoch 10), so the two size-aware models differ and can be fused.
# Synthetic-only: the initial weights were trained on synthetic data only, and so is this run.
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
repo="$HOME/projects/detector-program/OpenPCDet"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
init=$(find "$repo/output" -path "*/vnx010_all_s1234/ckpt/checkpoint_epoch_10.pth" -print -quit)
echo "init $init"
[ -f "$init" ] || { echo "NO_INIT_CKPT"; exit 1; }
export WORKERS=3
BS=2 bash "$here/run_arm_par_ns.sh" vnx010_pv_ft2 "$here/vnx010_all.yaml" "$D/claude_canon_all_twin_pv_v1" 5 "$G" --use_amp --pretrained_model "$init"
echo LANEP_DONE
