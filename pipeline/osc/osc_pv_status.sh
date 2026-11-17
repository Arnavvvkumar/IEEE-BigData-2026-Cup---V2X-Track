#!/bin/bash
# READ-ONLY: OSC PV-RCNN job state + training progress, then the laptop truck-voter test and the fourth fine-tune.
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E "$F" | cut -c1-230
source /etc/profile 2>/dev/null
squeue -u <osc_user> -o "%i %P %T %M %l %R"
squeue -j 8548278 --start 2>/dev/null | tail -n 1
R=/fs/scratch/<osc_project>/bdc
o=$(ls -t $R/logs/pvfull_*.out 2>/dev/null | head -n 1)
[ -n "$o" ] && { echo "--- $o"; grep -v -E 'it/s\]|s/it\]' "$o" | tail -n 3; }
t=$(ls -t $(find $R/code/OpenPCDet/output -path '*pvrcnn_st_osc*' -name 'train_*.log' 2>/dev/null) 2>/dev/null | head -n 1)
[ -n "$t" ] && { echo "--- train log"; grep -E 'Train:' "$t" | tail -n 2 | sed -E 's/.*Train: +/Train: /' | cut -c1-200; grep -E 'Error|Traceback|out of memory' "$t" | tail -n 2; }
EOF
echo "=== laptop"
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
grep -E '===|current|a 0.5 base|passes|P10_DONE' final/p10_vnx_truck.log | cut -c1-260
echo "ft4: $(grep -E 'Train:' ~/projects/detector-program/vnx010_pv_ft4.log | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-100)"
