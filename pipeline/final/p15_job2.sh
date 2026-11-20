#!/bin/bash
# P15 (laptop; OSC read-only): wait for job 2 (8560715, tag pvrcnn_st_osc_s2) to leave the queue, fetch its last checkpoint,
# run inference, and score it on top of v8 as a pedestrian (with hedging) and vehicle voter. Waits for p12 so evals run one at a time.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
TAG=pvrcnn_st_osc_s2
echo "P15_START $(date -Is)"
while ssh $K ${OSC_USER}@ascend.osc.edu 'source /etc/profile 2>/dev/null; squeue -h -j 8560715 2>/dev/null' 2>/dev/null | grep -q 8560715; do sleep 300; done
last=$(ssh $K ${OSC_USER}@ascend.osc.edu "find /fs/scratch/${OSC_PROJECT}/bdc/code/OpenPCDet/output -path '*/$TAG/ckpt/checkpoint_epoch_*.pth'" 2>/dev/null | sed -E 's/.*epoch_([0-9]+).pth/\1/' | sort -n | tail -n 1)
echo "job 2 finished; last checkpoint epoch $last $(date -Is)"
[ -n "$last" ] || { echo NO_CKPT; exit 1; }
bash "$base/osc/fetch_pv_ckpt.sh" "$last" "$TAG"
while pgrep -f p12_combine.sh > /dev/null; do sleep 60; done
BATCH=1 bash "$base/final/run_infer_one.sh" "$TAG" pvrcnn_all.yaml claude_canon_all_twin_st_v1 "$last"
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V8="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep2:pedestrian,vehicle"
for s in dev150 dev350; do
  echo "=== job 2 ($TAG ep$last) on $s $(date -Is)"
  # shellcheck disable=SC2086
  "$P" eval150.py $s v8s2_norule $V8 ${TAG}_ep$last:pedestrian,vehicle norule > /dev/null
  "$P" verify_hedge_fn.py $s v8s2_norule | tail -n 1
  "$P" rule_retune.py $s v8s2_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s v8 + job2: /"
done
echo "P15_DONE $(date -Is)"
