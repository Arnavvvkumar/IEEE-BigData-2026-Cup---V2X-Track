#!/bin/bash
# P28 (laptop; OSC read-only polling): when P4 (8585265), P5 (8585266), P6 (8585267) leave the queue, fetch the last checkpoint,
# infer dev (batch 1), per-scan test export (batch 1, fixed seed), then gate on top of the v13 list (B + P2 x3 + P3 x3): each new voter x1 / x3.
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian"
export FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle
JOBS=("8585266 pvrcnn_ped_P5 pvrcnn_ped.yaml claude_canon_all_v1" "8585267 pvrcnn_ped_P6 pvrcnn_ped_aug.yaml claude_canon_all_v1" "8585265 pvrcnn_ped_P4 pvrcnn_ped_aug.yaml claude_canon_all_twin_st_v1")
echo "P28_START $(date -Is)"
done_tags=""
for job in "${JOBS[@]}"; do
  set -- $job; jid=$1; tag=$2; cfg=$3; ds=$4
  while ssh $K <osc_user>@ascend.osc.edu "source /etc/profile 2>/dev/null; squeue -h -j $jid 2>/dev/null" 2>/dev/null | grep -q "$jid"; do sleep 300; done
  last=$(ssh $K <osc_user>@ascend.osc.edu "find /fs/scratch/<osc_project>/bdc/code/OpenPCDet/output -path '*/$tag/ckpt/checkpoint_epoch_*.pth'" 2>/dev/null | sed -E 's/.*epoch_([0-9]+).pth/\1/' | sort -n | tail -n 1)
  echo "$tag left queue; last epoch '$last' $(date -Is)"
  [ -n "$last" ] || { echo "NO_CKPT $tag"; continue; }
  bash "$base/osc/fetch_pv_ckpt.sh" "$last" "$tag" | tail -n 1
  BATCH=1 bash "$base/final/run_infer_one.sh" "$tag" "$cfg" "$ds" "$last" | tail -n 2
  # per-scan test export, batch 1, fixed seed (same wrapper as the final voters)
  sed -n '1,/^run() /p' "$base/final/reexport_perscan_b1.sh" > "$base/final/_reexport_$tag.sh"
  sed -n '/^run() /,/^}/p' "$base/final/reexport_perscan_b1.sh" | tail -n +2 >> "$base/final/_reexport_$tag.sh"
  echo "run $tag $cfg $ds $last 1" >> "$base/final/_reexport_$tag.sh"
  bash "$base/final/_reexport_$tag.sh" | grep -v '^skip'
  done_tags="$done_tags ${tag}_ep$last"
done
echo "P28_INFER_DONE $(date -Is) :$done_tags"
cd "$base/coop"
score() {  # set name passes
  "$P" eval150.py $1 $2 $3 norule > /dev/null
  "$P" verify_hedge_fn.py $1 $2 | tail -n 1 | sed "s/^/$1 $2 PED: /"
  "$P" rule_retune.py $1 $2 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$1 $2 VT: /"
}
for s in dev150 dev350; do
  score $s v13base_norule "$B"
  for t in $done_tags; do
    x1="$t:pedestrian"; x3="$t:pedestrian $t:pedestrian $t:pedestrian"
    score $s "v13_${t}_x1_norule" "$B $x1" &
    score $s "v13_${t}_x3_norule" "$B $x3" &
    wait
  done
done
echo "P28_DONE $(date -Is)"
