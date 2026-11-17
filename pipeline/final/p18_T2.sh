#!/bin/bash
# P18 (laptop; OSC read-only): when T2 (8563862, SECOND truck-only) leaves the queue, fetch its last checkpoint, infer, and score it on top
# of v9 (bundle B) as a truck voter with weight 1, 3 and 6 (the same pass listed several times = more votes), both dev sets.
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
TAG=second_truck_T2
echo "P18_START $(date -Is)"
while ssh $K <osc_user>@ascend.osc.edu 'source /etc/profile 2>/dev/null; squeue -h -j 8563862 2>/dev/null' 2>/dev/null | grep -q 8563862; do sleep 120; done
last=$(ssh $K <osc_user>@ascend.osc.edu "find /fs/scratch/<osc_project>/bdc/code/OpenPCDet/output -path '*/$TAG/ckpt/checkpoint_epoch_*.pth'" 2>/dev/null | sed -E 's/.*epoch_([0-9]+).pth/\1/' | sort -n | tail -n 1)
echo "T2 finished; last checkpoint epoch $last $(date -Is)"
[ -n "$last" ] || { echo NO_CKPT; exit 1; }
bash "$base/osc/fetch_pv_ckpt.sh" "$last" "$TAG"
bash "$base/final/run_infer_one.sh" "$TAG" second_truck.yaml claude_canon_all_twin_ll_v1 "$last"
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V9="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck"
T="${TAG}_ep$last:truck"
for s in dev150 dev350; do
  for w in 1 3 6; do
    x=$(for i in $(seq $w); do printf '%s ' "$T"; done)
    # shellcheck disable=SC2086
    "$P" eval150.py $s t2w${w}_norule $V9 $x norule > /dev/null
    "$P" rule_retune.py $s t2w${w}_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s + T2 truck voter x$w: /"
  done
done
echo "P18_DONE $(date -Is)"
