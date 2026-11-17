#!/bin/bash
# Levelled real-dev scoring of a trained arm. usage: eval_arm.sh <tag> <cfg> <dataset> "<epochs>" "<grounds>"
set -euo pipefail
tag=$1; config=$2; dataset=$3; eps=$4; grounds=$5
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
program="$HOME/projects/detector-program"
repo="$program/OpenPCDet"
res="$root/experiments/claude_carla_overnight/canon/results"
mkdir -p "$res"
ckdir=$(dirname "$(find "$repo/output" -path "*/$tag/ckpt/checkpoint_epoch_*.pth" -print -quit)")
cd "$repo/tools"
for e in $eps; do
  rm -f "$res/${tag}_ep${e}.json"
  "$program/env310/bin/python" "$root/experiments/claude_carla_overnight/erroranalysis/detilt_eval.py" --repo "$repo" --cfg "$config" --data "$dataset" \
    --checkpoint "$ckdir/checkpoint_epoch_${e}.pth" --out "$res/${tag}_ep${e}.json" --plane median --ground $grounds --save-best 2>&1 \
    | grep -E '^median|Error|Traceback' | sed "s/^/$tag ep$e /" | tee -a "$res/summary.txt"
done
