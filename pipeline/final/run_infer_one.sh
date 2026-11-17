#!/bin/bash
# Six-height levelled inference of ONE model on the validation sets and the released test scans (no labels). Skips done outputs.
# usage: run_infer_one.sh <tag> <cfg name in canon/> <dataset name> <epoch>
set -uo pipefail
tag=$1; cfg=$2; ds=$3; e=$4
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
base="$root/experiments/claude_carla_overnight"
program="$HOME/projects/detector-program"
repo="$program/OpenPCDet"
D="$HOME/projects/bdc2026-track-b/data"
G='-3.15 -3.075 -3.0 -2.925 -2.85 -2.775'
ckdir=$(dirname "$(find "$repo/output" -path "*/$tag/ckpt/checkpoint_epoch_*.pth" -print -quit)")
for pair in "$base/coop/dev150/scans|$base/coop/dev150/allpred" "$root/starting_kit/track_b/detection_test_frames|$base/final/test_allpred" \
            "$base/coop/dev350/scans|$base/coop/dev350/allpred"; do
  scans=${pair%%|*}; outdir=${pair##*|}
  [ -d "$scans" ] || continue
  out="$outdir/${tag}_ep${e}_allpred.json"
  [ -f "$out" ] && { echo "skip $out"; continue; }
  mkdir -p "$outdir"
  ( cd "$repo/tools" && "$program/env310/bin/python" "$base/final/infer_test.py" --repo "$repo" --cfg "$base/canon/$cfg" --data "$D/$ds" \
      --checkpoint "$ckdir/checkpoint_epoch_${e}.pth" --scans "$scans" --out "$out" --ground $G --batch "${BATCH:-4}" 2>&1 | grep -E '^scans|Error|Traceback' | sed "s/^/$tag /" )
  sleep 10
done
echo "INFER_ONE_DONE $tag"
