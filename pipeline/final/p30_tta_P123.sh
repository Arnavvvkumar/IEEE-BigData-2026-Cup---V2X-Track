#!/bin/bash
# P30: P1 TTA x3 passed (dev150 44.9->45.2 [44.8->45.0, 45.1->45.6]; dev350 44.5->44.9 [45.2->45.8, 44.0->44.2]; other classes identical).
# 1) test export of P1 TTA (per-scan, batch 1, fixed seed);  2) dev TTA for P2, P3;  3) gate cumulatively on top of v13 + P1tta x3;
# 4) test export of P2/P3 TTA.  Same variants as the VoxelNeXt TTA passes.
set -uo pipefail
root="${BDC_ROOT}"
base="$root/experiments/claude_carla_overnight"; canon="$base/canon"
program="${BDC_PROGRAM}"; repo="$program/OpenPCDet"; D="${BDC_DATA}"
P="$program/env310/bin/python"; scans="$root/starting_kit/track_b/detection_test_frames"; out="$base/final/test_allpred_ps1"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
export FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle
V="0.05,0.05,0 0.05,0,7 0,0.05,-7 0.025,0.075,14 0.075,0.025,-14"
ck() { echo "$(dirname "$(find "$repo/output" -path "*/$1/ckpt/checkpoint_epoch_*.pth" -print -quit)")/checkpoint_epoch_5.pth"; }
devtta() {  # tag
  for s in dev150 dev350; do
    o="$base/coop/$s/allpred/${1}_tta_ep5_allpred.json"; [ -f "$o" ] && continue
    ( cd "$repo/tools" && "$P" "$base/final/infer_tta.py" --repo "$repo" --cfg "$canon/pvrcnn_ped.yaml" --data "$D/claude_canon_all_twin_st_v1" \
        --checkpoint "$(ck $1)" --scans "$base/coop/$s/scans" --out "$o" --ground -3.075 -3.0 -2.925 -2.85 --variants $V --batch 2 --workers 6 2>&1 | grep -E 'Error|Traceback' )
    echo "$1 $s dev tta done $(date -Is)"
  done
}
testtta() {  # tag
  o="$out/${1}_tta_ep5_allpred.json"; [ -f "$o" ] && return
  ( cd "$repo/tools" && PLANE=scan "$P" "$base/codex_review/per_scan_export.py" infer_tta.py --repo "$repo" --cfg "$canon/pvrcnn_ped.yaml" --data "$D/claude_canon_all_twin_st_v1" \
      --checkpoint "$(ck $1)" --scans "$scans" --out "$o" --ground -3.075 -3.0 -2.925 -2.85 --variants $V --batch 8 --workers 6 2>&1 | grep -E '^plane mode|Error|Traceback' )
  echo "$1 test tta done $(date -Is)"
}
score() {
  "$P" eval150.py $1 $2 $3 norule > /dev/null
  "$P" verify_hedge_fn.py $1 $2 | tail -n 1 | sed "s/^/$1 $2 PED: /"
  "$P" rule_retune.py $1 $2 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$1 $2 VT: /"
}
echo "P30_START $(date -Is)"
testtta pvrcnn_ped_P1
devtta pvrcnn_ped_P2
devtta pvrcnn_ped_P3
B1="$B $(x3 pvrcnn_ped_P1_tta_ep5)"
cd "$base/coop"
for s in dev150 dev350; do
  score $s v14_p1t_norule "$B1" &
  score $s v14_p1t_p2t_norule "$B1 $(x3 pvrcnn_ped_P2_tta_ep5)" &
  wait
  score $s v14_p1t_p2t_p3t_norule "$B1 $(x3 pvrcnn_ped_P2_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5)" &
  score $s v14_p1t_p3t_norule "$B1 $(x3 pvrcnn_ped_P3_tta_ep5)" &
  wait
done
echo "P30_GATE_DONE $(date -Is)"
testtta pvrcnn_ped_P2
testtta pvrcnn_ped_P3
echo "P30_DONE $(date -Is)"
