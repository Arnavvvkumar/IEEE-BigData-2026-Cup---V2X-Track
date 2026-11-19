#!/bin/bash
# Compliance re-export: every final voter re-run on the 50 released test scans with PER-SCAN ground levelling (PLANE=scan):
# each scan is levelled only with the plane fitted to its own points; no other test scan is used. Output: final/test_allpred_ps/.
# The long-low truck voter (second_twin_ll_*) is excluded (its data recipe referenced test-clip objects).
set -uo pipefail
root="${BDC_ROOT}"
base="$root/experiments/claude_carla_overnight"; here="$base/final"; canon="$base/canon"
program="${BDC_PROGRAM}"; repo="$program/OpenPCDet"; D="${BDC_DATA}"
scans="$root/starting_kit/track_b/detection_test_frames"; out="$here/test_allpred_ps"
G='-3.15 -3.075 -3.0 -2.925 -2.85 -2.775'
export PLANE=scan
mkdir -p "$out"
echo "REEXPORT_START $(date -Is)"
run() {  # tag cfg dataset epoch batch
  o="$out/${1}_ep${4}_allpred.json"; [ -f "$o" ] && { echo "skip $1"; return; }
  ck=$(dirname "$(find "$repo/output" -path "*/$1/ckpt/checkpoint_epoch_*.pth" -print -quit)")/checkpoint_epoch_${4}.pth
  [ -f "$ck" ] || { echo "MISSING_CKPT $1 $ck"; return; }
  ( cd "$repo/tools" && "$program/env310/bin/python" "$here/infer_test.py" --repo "$repo" --cfg "$canon/$2" --data "$D/$3" \
      --checkpoint "$ck" --scans "$scans" --out "$o" --ground $G --batch "$5" 2>&1 | grep -E '^plane mode|Error|Traceback' | sed "s/^/$1 /" )
  echo "done $1 $(date +%H:%M:%S)"
}
tta() {  # tag cfg dataset epoch
  o="$out/${1}_tta_ep${4}_allpred.json"; [ -f "$o" ] && { echo "skip $1 tta"; return; }
  ck=$(dirname "$(find "$repo/output" -path "*/$1/ckpt/checkpoint_epoch_*.pth" -print -quit)")/checkpoint_epoch_${4}.pth
  ( cd "$repo/tools" && "$program/env310/bin/python" "$here/infer_tta.py" --repo "$repo" --cfg "$canon/$2" --data "$D/$3" \
      --checkpoint "$ck" --scans "$scans" --out "$o" --ground -3.075 -3.0 -2.925 -2.85 \
      --variants 0.05,0.05,0 0.05,0,7 0,0.05,-7 0.025,0.075,14 0.075,0.025,-14 --batch 8 --workers 8 2>&1 | grep -E '^plane mode|Error|Traceback' | sed "s/^/$1 tta /" )
  echo "done $1 tta $(date +%H:%M:%S)"
}
# fast SECOND / VoxelNeXt voters first
run vnx010_all_s1234     vnx010_all.yaml           claude_canon_all_v1              10 4
run vnx010_twin_s1       vnx010_all.yaml           claude_canon_all_twin_v1         12 4
run vnx010_all_rs2_amp   vnx010_all.yaml           claude_canon_all_v1              12 4
run vnx010_pv_ft         vnx010_all.yaml           claude_canon_all_twin_pv_v1       5 1
run vnx010_pv_ft2        vnx010_all.yaml           claude_canon_all_twin_pv_v1       5 1
run vnx010_pv_ft4        vnx010_all.yaml           claude_canon_all_twin_pv_v1       4 1
run canon_all_s1234      second_canon_all_v1.yaml  claude_canon_all_v1              12 4
run second_twin_s1       second_canon_all_v1.yaml  claude_canon_all_twin_v1         12 4
run canon_relabel_s1234  second_canon_v1.yaml      claude_canon_relabel_mix6000_v1  12 4
run second_twin_s2       second_canon_all_v1.yaml  claude_canon_all_twin_v1         12 4
run second_twin_st_s1    second_canon_all_v1.yaml  claude_canon_all_twin_st_v1      12 4
run second_twin_pv_s1    second_canon_all_v1.yaml  claude_canon_all_twin_pv_v1      14 4
# PV-RCNN voters
run pvrcnn_st_osc        pvrcnn_all.yaml           claude_canon_all_twin_st_v1       6 1
run pvrcnn_st_osc_s2     pvrcnn_all.yaml           claude_canon_all_twin_st_v1       4 1
run pvrcnn_ped_P1        pvrcnn_ped.yaml           claude_canon_all_twin_st_v1       5 1
run pvrcnn_ped_P2        pvrcnn_ped.yaml           claude_canon_all_twin_st_v1       5 1
run pvrcnn_veh_V1        pvrcnn_veh.yaml           claude_canon_all_twin_st_v1       5 1
# test-time shift/rotation passes
tta vnx010_twin_s1       vnx010_all.yaml           claude_canon_all_twin_v1         12
tta vnx010_all_s1234     vnx010_all.yaml           claude_canon_all_v1              10
ls "$out" | wc -l
echo "REEXPORT_DONE $(date -Is)"
