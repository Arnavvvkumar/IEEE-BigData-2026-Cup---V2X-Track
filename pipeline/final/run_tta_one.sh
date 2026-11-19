#!/bin/bash
# Shift/rotation test-time passes of ONE model on one scan folder. Output tag gets the suffix "_tta".
# usage: run_tta_one.sh <tag> <cfg name in canon/> <dataset name> <epoch> <scans dir> <out dir>
set -uo pipefail
tag=$1; cfg=$2; ds=$3; e=$4; scans=$5; outdir=$6
root="${BDC_ROOT}"
base="$root/experiments/claude_carla_overnight"
program="${BDC_PROGRAM}"
repo="$program/OpenPCDet"
D="${BDC_DATA}"
ckdir=$(dirname "$(find "$repo/output" -path "*/$tag/ckpt/checkpoint_epoch_*.pth" -print -quit)")
out="$outdir/${tag}_tta_ep${e}_allpred.json"
[ -f "$out" ] && { echo "skip $out"; exit 0; }
mkdir -p "$outdir"
cd "$repo/tools"
"$program/env310/bin/python" "$base/final/infer_tta.py" --repo "$repo" --cfg "$base/canon/$cfg" --data "$D/$ds" \
  --checkpoint "$ckdir/checkpoint_epoch_${e}.pth" --scans "$scans" --out "$out" --ground -3.075 -3.0 -2.925 -2.85 \
  --variants 0.05,0.05,0 0.05,0,7 0,0.05,-7 0.025,0.075,14 0.075,0.025,-14 --batch 32 --workers 8 2>&1 | stdbuf -oL grep -E '^scans|^pass|^batch|Error|Traceback' | stdbuf -oL sed "s/^/$tag /"
echo "TTA_DONE $tag"
