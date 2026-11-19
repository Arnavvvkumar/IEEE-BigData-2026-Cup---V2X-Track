#!/bin/bash
# Compliance re-export: every final voter re-run on the 50 released test scans with PER-SCAN ground levelling (PLANE=scan):
# each scan is levelled only with the plane fitted to its own points; no other test scan is used. Output: final/test_allpred_ps/.
# The long-low truck voter (second_twin_ll_*) is excluded (its data recipe referenced test-clip objects).
set -uo pipefail
root="${BDC_ROOT}"
base="$root/experiments/claude_carla_overnight"; here="$base/final"; canon="$base/canon"
program="${BDC_PROGRAM}"; repo="$program/OpenPCDet"; D="${BDC_DATA}"
scans="$root/starting_kit/track_b/detection_test_frames"; out="$here/test_allpred_ps1"
G='-3.15 -3.075 -3.0 -2.925 -2.85 -2.775'
export PLANE=scan
mkdir -p "$out"
echo "REEXPORT1_START $(date -Is)"
run() {  # tag cfg dataset epoch batch
  o="$out/${1}_ep${4}_allpred.json"; [ -f "$o" ] && { echo "skip $1"; return; }
  ck=$(dirname "$(find "$repo/output" -path "*/$1/ckpt/checkpoint_epoch_*.pth" -print -quit)")/checkpoint_epoch_${4}.pth
  [ -f "$ck" ] || { echo "MISSING_CKPT $1 $ck"; return; }
  ( cd "$repo/tools" && "$program/env310/bin/python" "$base/codex_review/per_scan_export.py" infer_test.py --repo "$repo" --cfg "$canon/$2" --data "$D/$3" \
      --checkpoint "$ck" --scans "$scans" --out "$o" --ground $G --batch "$5" 2>&1 | grep -E '^plane mode|Error|Traceback' | sed "s/^/$1 /" )
  echo "done $1 $(date +%H:%M:%S)"
}
run vrcnn_ped_VR2 voxelrcnn_ped.yaml claude_canon_all_v1 5 1
