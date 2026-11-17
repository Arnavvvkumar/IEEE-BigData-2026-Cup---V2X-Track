#!/bin/bash
# P34: P4 TTA (P4 x1 passed on v14s). Dev150/dev350 TTA, then test export (per-scan, batch fixed), same variants as P1/P3 TTA.
# Gate afterwards: v15s list + P4tta x1 / x3 (pedestrian).
set -uo pipefail
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
base="$root/experiments/claude_carla_overnight"; canon="$base/canon"
program="$HOME/projects/detector-program"; repo="$program/OpenPCDet"; D="$HOME/projects/bdc2026-track-b/data"
P="$program/env310/bin/python"; scans="$root/starting_kit/track_b/detection_test_frames"; out="$base/final/test_allpred_ps1"
V="0.05,0.05,0 0.05,0,7 0,0.05,-7 0.025,0.075,14 0.075,0.025,-14"
CFG="$canon/pvrcnn_ped_aug.yaml"; DATA="$D/claude_canon_all_twin_st_v1"
CK="$(dirname "$(find "$repo/output" -path '*/pvrcnn_ped_P4/ckpt/checkpoint_epoch_*.pth' -print -quit)")/checkpoint_epoch_5.pth"
echo "P34_START $(date -Is) ckpt $CK"; ls -la "$CK" || exit 1
for s in dev150 dev350; do
  o="$base/coop/$s/allpred/pvrcnn_ped_P4_tta_ep5_allpred.json"; [ -f "$o" ] && continue
  ( cd "$repo/tools" && "$P" "$base/final/infer_tta.py" --repo "$repo" --cfg "$CFG" --data "$DATA" --checkpoint "$CK" --scans "$base/coop/$s/scans" \
      --out "$o" --ground -3.075 -3.0 -2.925 -2.85 --variants $V --batch 2 --workers 6 2>&1 | grep -E 'Error|Traceback' )
  echo "P4 $s dev tta done $(date -Is)"
done
o="$out/pvrcnn_ped_P4_tta_ep5_allpred.json"
[ -f "$o" ] || ( cd "$repo/tools" && PLANE=scan "$P" "$base/codex_review/per_scan_export.py" infer_tta.py --repo "$repo" --cfg "$CFG" --data "$DATA" \
    --checkpoint "$CK" --scans "$scans" --out "$o" --ground -3.075 -3.0 -2.925 -2.85 --variants $V --batch 8 --workers 6 2>&1 | grep -E '^plane mode|Error|Traceback' )
echo "P4 test tta done $(date -Is)"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
B="$B $(x3 pvrcnn_ped_P1_ep5) $(x3 pvrcnn_ped_P2_ep5) $(x3 pvrcnn_ped_P3_ep5) $(x3 pvrcnn_ped_P1_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5) pvrcnn_ped_P4_ep5:pedestrian"
export FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle
cd "$base/coop"
ped() { "$P" eval150.py $1 $2 $3 norule > /dev/null; "$P" verify_hedge_fn.py $1 $2 | tail -n 1 | sed "s/^/$1 $2 PED: /"; }
for s in dev150 dev350; do
  ped $s v15_p4t1_norule "$B pvrcnn_ped_P4_tta_ep5:pedestrian"
  ped $s v15_p4t3_norule "$B $(x3 pvrcnn_ped_P4_tta_ep5)"
done
echo "P34_DONE $(date -Is)"
