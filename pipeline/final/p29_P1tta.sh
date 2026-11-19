#!/bin/bash
# P29 (laptop GPU, idle until P4-P6 return): Codex rank-2 idea - test-time shift/rotation agreement for an existing PV-RCNN
# pedestrian checkpoint (P1), exactly like the VoxelNeXt TTA passes already in the final list. Dev only; gate on top of v13 list.
set -uo pipefail
root="${BDC_ROOT}"
base="$root/experiments/claude_carla_overnight"; canon="$base/canon"
program="${BDC_PROGRAM}"; repo="$program/OpenPCDet"; D="${BDC_DATA}"
P="$program/env310/bin/python"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian"
export FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle
tag=pvrcnn_ped_P1; e=5
ck=$(dirname "$(find "$repo/output" -path "*/$tag/ckpt/checkpoint_epoch_*.pth" -print -quit)")/checkpoint_epoch_$e.pth
echo "P29_START $(date -Is) $ck"
for s in dev150 dev350; do
  o="$base/coop/$s/allpred/${tag}_tta_ep${e}_allpred.json"; [ -f "$o" ] && continue
  ( cd "$repo/tools" && "$P" "$base/final/infer_tta.py" --repo "$repo" --cfg "$canon/pvrcnn_ped.yaml" --data "$D/claude_canon_all_twin_st_v1" \
      --checkpoint "$ck" --scans "$base/coop/$s/scans" --out "$o" --ground -3.075 -3.0 -2.925 -2.85 \
      --variants 0.05,0.05,0 0.05,0,7 0,0.05,-7 0.025,0.075,14 0.075,0.025,-14 --batch 2 --workers 6 2>&1 | grep -E '^scans|Error|Traceback' )
  echo "$s tta done $(date -Is)"
done
cd "$base/coop"
score() {
  "$P" eval150.py $1 $2 $3 norule > /dev/null
  "$P" verify_hedge_fn.py $1 $2 | tail -n 1 | sed "s/^/$1 $2 PED: /"
  "$P" rule_retune.py $1 $2 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$1 $2 VT: /"
}
T="${tag}_tta_ep$e:pedestrian"
for s in dev150 dev350; do
  score $s v13base_norule "$B" &
  score $s v13_P1tta_x1_norule "$B $T" &
  wait
  score $s v13_P1tta_x3_norule "$B $T $T $T"
done
echo "P29_DONE $(date -Is)"
