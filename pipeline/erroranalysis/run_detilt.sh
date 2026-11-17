#!/bin/bash
# usage: run_detilt.sh <name> <plane mode> <ground z ...>   (SECOND mix epoch 12)
set -euo pipefail
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
program="$HOME/projects/detector-program"
name=$1; mode=$2; shift 2
cd "$program/OpenPCDet/tools"
"$program/env310/bin/python" "$root/experiments/claude_carla_overnight/erroranalysis/detilt_eval.py" --repo "$program/OpenPCDet" \
  --cfg "$root/experiments/detector_program/second_control_v1.yaml" --data "$HOME/projects/bdc2026-track-b/data/claude_v2carla_mix6000_v1" \
  --checkpoint "$program/OpenPCDet/output/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/detector_program/second_control_v1/overnight_mix_seed1234/ckpt/checkpoint_epoch_12.pth" \
  --out "$root/experiments/claude_carla_overnight/erroranalysis/detilt_$name.json" --plane "$mode" --ground "$@" --save-best 2>&1 | grep -E 'plane|ground|Error|error|Traceback'
