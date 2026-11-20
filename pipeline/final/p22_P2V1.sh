#!/bin/bash
# P22 (laptop; OSC read-only): when P2 (8571774, pedestrian-only PV-RCNN seed 2) and V1 (8571775, vehicle-only PV-RCNN) leave the queue,
# fetch their last checkpoints, infer (batch 1), and score on top of v10 (v9 + P1 x3 + vehicle one-vote):
#   P2 as pedestrian voter x1/x3 (with hedging);  V1 as vehicle voter x1/x3 (vehicle AP and truck AP via the v5 truck rule).
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
V10="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian"
export FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle
echo "P22_START $(date -Is)"
for job in "8571774 pvrcnn_ped_P2 pvrcnn_ped.yaml pedestrian" "8571775 pvrcnn_veh_V1 pvrcnn_veh.yaml vehicle"; do
  set -- $job; jid=$1; tag=$2; cfg=$3; cls=$4
  while ssh $K ${OSC_USER}@ascend.osc.edu "source /etc/profile 2>/dev/null; squeue -h -j $jid 2>/dev/null" 2>/dev/null | grep -q "$jid"; do sleep 300; done
  last=$(ssh $K ${OSC_USER}@ascend.osc.edu "find /fs/scratch/${OSC_PROJECT}/bdc/code/OpenPCDet/output -path '*/$tag/ckpt/checkpoint_epoch_*.pth'" 2>/dev/null | sed -E 's/.*epoch_([0-9]+).pth/\1/' | sort -n | tail -n 1)
  echo "$tag finished; last epoch $last $(date -Is)"
  [ -n "$last" ] || { echo "NO_CKPT $tag"; continue; }
  bash "$base/osc/fetch_pv_ckpt.sh" "$last" "$tag"
  BATCH=1 bash "$base/final/run_infer_one.sh" "$tag" "$cfg" claude_canon_all_twin_st_v1 "$last"
  cd "$base/coop"
  for s in dev150 dev350; do
    for w in 1 3; do
      x=$(for i in $(seq $w); do printf '%s ' "${tag}_ep$last:$cls"; done)
      # shellcheck disable=SC2086
      "$P" eval150.py $s ${tag}w${w}_norule $V10 $x norule > /dev/null
      if [ $cls = pedestrian ]; then
        "$P" verify_hedge_fn.py $s ${tag}w${w}_norule | tail -n 1 | sed "s/^/x$w /"
      else
        "$P" rule_retune.py $s ${tag}w${w}_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s + $tag x$w: /"
      fi
    done
  done
done
echo "P22_DONE $(date -Is)"
