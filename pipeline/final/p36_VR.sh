#!/bin/bash
# P36 (laptop; OSC read-only polling): when VR2 (8594089) and VR1 (8594088) leave the queue, fetch the last checkpoint, infer dev150/dev350
# (batch 1), per-scan test export (batch 1, fixed seed), then gate pedestrian AP on top of the v16 list (v15 list + averaged pedestrian heading).
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
CFG=voxelrcnn_ped.yaml
JOBS=("8594089 vrcnn_ped_VR2 claude_canon_all_v1" "8594088 vrcnn_ped_VR1 claude_canon_all_twin_st_v1")
echo "P36_START $(date -Is)"
done_tags=""
for job in "${JOBS[@]}"; do
  set -- $job; jid=$1; tag=$2; ds=$3
  while ssh $K ${OSC_USER}@ascend.osc.edu "source /etc/profile 2>/dev/null; squeue -h -j $jid 2>/dev/null" 2>/dev/null | grep -q "$jid"; do sleep 300; done
  last=$(ssh $K ${OSC_USER}@ascend.osc.edu "find /fs/scratch/${OSC_PROJECT}/bdc/code/OpenPCDet/output -path '*/$tag/ckpt/checkpoint_epoch_*.pth'" 2>/dev/null | sed -E 's/.*epoch_([0-9]+).pth/\1/' | sort -n | tail -n 1)
  echo "$tag left queue; last epoch '$last' $(date -Is)"
  [ -n "$last" ] || { echo "NO_CKPT $tag"; continue; }
  bash "$base/osc/fetch_pv_ckpt.sh" "$last" "$tag" | tail -n 1
  BATCH=1 bash "$base/final/run_infer_one.sh" "$tag" "$CFG" "$ds" "$last" | tail -n 2
  sed -n '1,/^run() /p' "$base/final/reexport_perscan_b1.sh" > "$base/final/_reexport_$tag.sh"
  sed -n '/^run() /,/^}/p' "$base/final/reexport_perscan_b1.sh" | tail -n +2 >> "$base/final/_reexport_$tag.sh"
  echo "run $tag $CFG $ds $last 1" >> "$base/final/_reexport_$tag.sh"
  bash "$base/final/_reexport_$tag.sh" | grep -v '^skip'
  done_tags="$done_tags ${tag}_ep$last"
done
echo "P36_INFER_DONE $(date -Is) :$done_tags"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
B="$B $(x3 pvrcnn_ped_P1_ep5) $(x3 pvrcnn_ped_P2_ep5) $(x3 pvrcnn_ped_P3_ep5) $(x3 pvrcnn_ped_P1_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5) pvrcnn_ped_P4_ep5:pedestrian"
export FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle
cd "$base/coop"
ped() { "$P" eval150.py $1 $2 $3 norule > /dev/null; "$P" verify_hedge_fn.py $1 $2 | tail -n 1 | sed "s/^/$1 $2 PED: /"; }
set -- $done_tags; a=${1:-}; b=${2:-}
for s in dev150 dev350; do
  ped $s v16base_norule "$B"
  for t in $done_tags; do
    ped $s "v16_${t}_x1_norule" "$B $t:pedestrian"
    ped $s "v16_${t}_x3_norule" "$B $(x3 $t)"
  done
  if [ -n "$b" ]; then
    ped $s v16_VRboth_x1_norule "$B $a:pedestrian $b:pedestrian"
    ped $s v16_VRboth_x3_norule "$B $(x3 $a) $(x3 $b)"
  fi
done
echo "P36_DONE $(date -Is)"
