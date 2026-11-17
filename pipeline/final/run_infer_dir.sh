#!/bin/bash
# Six-height levelled inference of the ensemble members on any folder of xyz scans (no labels). Skips members already done.
# usage: run_infer_dir.sh <scans dir> <out dir>
set -uo pipefail
scans=$1; outdir=$2
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
here="$root/experiments/claude_carla_overnight/final"
canon="$root/experiments/claude_carla_overnight/canon"
program="$HOME/projects/detector-program"
repo="$program/OpenPCDet"
D="$HOME/projects/bdc2026-track-b/data"
G='-3.15 -3.075 -3.0 -2.925 -2.85 -2.775'
mkdir -p "$outdir"
run() {  # tag cfg dataset epoch
  out="$outdir/${1}_ep${4}_allpred.json"
  [ -f "$out" ] && { echo "skip $1"; return; }
  ckdir=$(dirname "$(find "$repo/output" -path "*/$1/ckpt/checkpoint_epoch_*.pth" -print -quit)")
  ( cd "$repo/tools" && "$program/env310/bin/python" "$here/infer_test.py" --repo "$repo" --cfg "$2" --data "$3" \
      --checkpoint "$ckdir/checkpoint_epoch_${4}.pth" --scans "$scans" --out "$out" --ground $G --batch 16 --workers 8 2>&1 | stdbuf -oL grep -E '^scans|^ground|^batch|Error|Traceback' | stdbuf -oL sed "s/^/$1 /" )
  sleep 5
}
run vnx010_all_s1234   "$canon/vnx010_all.yaml"           "$D/claude_canon_all_v1"      10
run vnx010_twin_s1     "$canon/vnx010_all.yaml"           "$D/claude_canon_all_twin_v1" 12
run canon_all_s1234    "$canon/second_canon_all_v1.yaml"  "$D/claude_canon_all_v1"      12
run second_twin_s1     "$canon/second_canon_all_v1.yaml"  "$D/claude_canon_all_twin_v1" 12
run canon_relabel_s1234 "$canon/second_canon_v1.yaml"     "$D/claude_canon_relabel_mix6000_v1" 12
run vnx010_all_rs2_amp "$canon/vnx010_all.yaml"           "$D/claude_canon_all_v1"      12
echo INFER_DIR_DONE
