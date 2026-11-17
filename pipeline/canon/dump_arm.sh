#!/bin/bash
# Dump levelled real-dev predictions at several ground heights for fusion. usage: dump_arm.sh <tag> <cfg> <dataset> <epoch> "<grounds>"
set -euo pipefail
tag=$1; config=$2; dataset=$3; e=$4; grounds=$5
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
program="$HOME/projects/detector-program"
repo="$program/OpenPCDet"
res="$root/experiments/claude_carla_overnight/canon/fusion"
mkdir -p "$res"
ckdir=$(dirname "$(find "$repo/output" -path "*/$tag/ckpt/checkpoint_epoch_*.pth" -print -quit)")
cd "$repo/tools"
"$program/env310/bin/python" "$root/experiments/claude_carla_overnight/erroranalysis/detilt_eval.py" --repo "$repo" --cfg "$config" --data "$dataset" \
  --checkpoint "$ckdir/checkpoint_epoch_${e}.pth" --out "$res/${tag}_ep${e}.json" --plane median --ground $grounds --save-all 2>&1 \
  | grep -E '^median|Error|Traceback' | sed "s/^/$tag ep$e /" | tee -a "$res/dump_summary.txt"
