#!/bin/bash
# Six-height levelled inference on the 50 released test scans (no labels) for each ensemble member. Skips members already done.
# usage: run_test_infer.sh  (edit nothing while it runs)
set -uo pipefail
root="${BDC_ROOT}"
here="$root/experiments/claude_carla_overnight/final"
canon="$root/experiments/claude_carla_overnight/canon"
program="${BDC_PROGRAM}"
repo="$program/OpenPCDet"
D="${BDC_DATA}"
scans="$root/starting_kit/track_b/detection_test_frames"
G='-3.15 -3.075 -3.0 -2.925 -2.85 -2.775'
mkdir -p "$here/test_allpred"
run() {  # tag cfg dataset epoch
  out="$here/test_allpred/${1}_ep${4}_allpred.json"
  [ -f "$out" ] && { echo "skip $1"; return; }
  ckdir=$(dirname "$(find "$repo/output" -path "*/$1/ckpt/checkpoint_epoch_*.pth" -print -quit)")
  ( cd "$repo/tools" && "$program/env310/bin/python" "$here/infer_test.py" --repo "$repo" --cfg "$2" --data "$3" \
      --checkpoint "$ckdir/checkpoint_epoch_${4}.pth" --scans "$scans" --out "$out" --ground $G 2>&1 | grep -E '^scans|^ground|Error|Traceback' | sed "s/^/$1 /" )
  sleep 20
}
run vnx010_all_s1234   "$canon/vnx010_all.yaml"           "$D/claude_canon_all_v1"      10
run vnx010_all_rs2_amp "$canon/vnx010_all.yaml"           "$D/claude_canon_all_v1"      12
run canon_all_s1234    "$canon/second_canon_all_v1.yaml"  "$D/claude_canon_all_v1"      12
run second_twin_s1     "$canon/second_canon_all_v1.yaml"  "$D/claude_canon_all_twin_v1" 12
run canon_all_rs2      "$canon/second_canon_all_v1.yaml"  "$D/claude_canon_all_v1"      12
run canon_relabel_s1234 "$canon/second_canon_v1.yaml"     "$D/claude_canon_relabel_mix6000_v1" 12
echo TEST_INFER_DONE
