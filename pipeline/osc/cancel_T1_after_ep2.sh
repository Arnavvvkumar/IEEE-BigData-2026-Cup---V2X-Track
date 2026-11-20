#!/bin/bash
# Approved by the user on 2026-11-13 ("after epoch 2 cancel"): wait for T1's epoch-2 checkpoint on OSC, copy it, then scancel T1 (8563861)
# and ONLY T1 (P1 = 8564353 keeps running). Polls read-only every 60 s, up to 45 minutes.
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
for i in $(seq 45); do
  n=$(ssh $K ${OSC_USER}@ascend.osc.edu "ls /fs/scratch/${OSC_PROJECT}/bdc/code/OpenPCDet/output/fs/scratch/${OSC_PROJECT}/bdc/cfg/pvrcnn_truck/pvrcnn_truck_T1/ckpt/checkpoint_epoch_2.pth 2>/dev/null | wc -l" 2>/dev/null | tail -n 1)
  [ "$n" = 1 ] && break
  sleep 60
done
echo "epoch-2 checkpoint present: $n after $i checks $(date -Is)"
[ "$n" = 1 ] || { echo "NOT_CANCELLED: epoch-2 checkpoint did not appear in time; T1 left running"; exit 1; }
sleep 20   # let the file finish writing
bash "${BDC_ROOT}/experiments/claude_carla_overnight/osc/fetch_pv_ckpt.sh" 2 pvrcnn_truck_T1
ssh $K ${OSC_USER}@ascend.osc.edu 'source /etc/profile 2>/dev/null; scancel 8563861; sleep 5; squeue -u ${OSC_USER} -o "%i %j %T %M"; sacct -j 8563861 -X -n --format=JobID,State,Elapsed' 2>&1 | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
echo "T1 CANCELLED $(date -Is)"
