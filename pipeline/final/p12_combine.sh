#!/bin/bash
# P12 (laptop; OSC only read-only: squeue + scp of the approved job's checkpoints). Scores, on top of v7:
#   A) PV-RCNN epoch 2 (early check)   B) long-low truck seed 2   C) PV-RCNN final epoch (after the OSC job ends)
# Each model is tried as pedestrian voter (scored WITH hedging), vehicle voter and truck voter. One eval at a time.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40"
CK="${BDC_PROGRAM}/OpenPCDet/output${BDC_ROOT}/experiments/claude_carla_overnight/canon/pvrcnn_all/pvrcnn_st_osc/ckpt"
V7="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle"
export FUSE_YAW_CIRC=vehicle

score() {  # score <tag_epN> <short name>
  local m=$1 n=$2
  cd "$base/coop"
  for s in dev150 dev350; do
    echo "=== $n on $s $(date -Is)"
    for use in pedestrian vehicle truck; do
      # shellcheck disable=SC2086
      "$P" eval150.py $s ${n}_${use}_norule $V7 $m:$use norule > /dev/null
      if [ $use = pedestrian ]; then
        "$P" verify_hedge_fn.py $s ${n}_${use}_norule | tail -n 1
      else
        "$P" rule_retune.py $s ${n}_${use}_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$use voter: /"
      fi
    done
  done
}

echo "P12_START $(date -Is)"
while pgrep -f p11_eval_ll.sh > /dev/null; do sleep 30; done
cd "$base/coop"
for s in dev150 dev350; do   # the v7 baseline itself, hedged pedestrians
  "$P" verify_hedge_fn.py $s v7_norule | tail -n 1
  "$P" rule_retune.py $s v7_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed 's/^/v7 baseline: /'
done

# A) PV-RCNN epoch 2 (the copy started at about 21:55)
while pgrep -f 'scp .*checkpoint_epoch_2' > /dev/null; do sleep 30; done
if [ -s "$CK/checkpoint_epoch_2.pth" ]; then
  BATCH=1 bash "$base/final/run_infer_one.sh" pvrcnn_st_osc pvrcnn_all.yaml claude_canon_all_twin_st_v1 2
  score pvrcnn_st_osc_ep2 pv2
fi

# B) long-low truck seed 2
until grep -q LANES2_DONE "$base/canon/laneS2.log" 2>/dev/null; do sleep 60; done
bash "$base/final/run_infer_one.sh" second_twin_ll_s2 second_canon_all_v1.yaml claude_canon_all_twin_ll_v1 12
cd "$base/coop"
for s in dev150 dev350; do
  echo "=== ll seed 2 on $s"
  # shellcheck disable=SC2086
  "$P" eval150.py $s ll2_norule $V7 second_twin_ll_s2_ep12:truck norule > /dev/null
  "$P" rule_retune.py $s ll2_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed 's/^/+ll seed 2 truck voter: /'
done

# C) PV-RCNN final epoch, once the approved OSC job 8548278 has left the queue (read-only polling, every 5 min)
while ssh $K ${OSC_USER}@ascend.osc.edu 'source /etc/profile 2>/dev/null; squeue -h -j 8548278 2>/dev/null' 2>/dev/null | grep -q 8548278; do sleep 300; done
last=$(ssh $K ${OSC_USER}@ascend.osc.edu "find /fs/scratch/${OSC_PROJECT}/bdc/code/OpenPCDet/output -path '*pvrcnn_st_osc*' -name 'checkpoint_epoch_*.pth'" 2>/dev/null | sed -E 's/.*epoch_([0-9]+).pth/\1/' | sort -n | tail -n 1)
echo "OSC job finished; last checkpoint epoch $last $(date -Is)"
if [ -n "$last" ] && [ "$last" != 2 ]; then
  bash "$base/osc/fetch_pv_ckpt.sh" "$last"
  BATCH=1 bash "$base/final/run_infer_one.sh" pvrcnn_st_osc pvrcnn_all.yaml claude_canon_all_twin_st_v1 "$last"
  score "pvrcnn_st_osc_ep$last" "pv$last"
fi
echo "P12_DONE $(date -Is)"
