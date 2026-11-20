#!/bin/bash
# Read-only: what survived the sleep? Local processes, log tails, OSC jobs.
date
echo "uptime: $(uptime -p)"
for p in p12_combine p15_job2 realism_swap realism_exp2 realism_exp3 eval150 infer_test 'scp ' globusconnect; do echo "$p: $(pgrep -fc "$p")"; done
b="${BDC_ROOT}/experiments/claude_carla_overnight"
for f in final/p12_combine.log final/p15_job2.log twin/realism_swap.log twin/realism_exp2.log twin/realism_exp3.log; do echo "--- $f"; tail -n 3 "$b/$f" 2>/dev/null | cut -c1-200; done
ls -la ${BDC_PROGRAM}/OpenPCDet/output${BDC_OUT}/'BIG DATA IEEE'/experiments/claude_carla_overnight/canon/pvrcnn_all/*/ckpt/ 2>/dev/null | grep pth
echo "--- OSC"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60 ${OSC_USER}@ascend.osc.edu 'source /etc/profile 2>/dev/null; squeue -u ${OSC_USER} -o "%i %T %M %l"; sacct -j 8548278,8560715 -X -n --format=JobID,State,Elapsed; find /fs/scratch/${OSC_PROJECT}/bdc/code/OpenPCDet/output -name "checkpoint_epoch_*.pth" -path "*pvrcnn_st_osc*" | sed -E "s|.*/(pvrcnn_st_osc[_s2]*)/ckpt/|\1 |"' 2>&1 | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-160
