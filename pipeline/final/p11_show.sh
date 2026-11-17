#!/bin/bash
# Status after the pause: P11 (long-low truck scoring), seed 2 training, OSC job (read-only), Globus.
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
date
grep -E 'P11_|===|passes|a 0.5 base|Traceback|Errno|INFER' "$base/final/p11_eval_ll.log" | cut -c1-260
echo "--- seed 2"
grep -E 'Train:' ~/projects/detector-program/second_twin_ll_s2.log | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-110
tail -n 5 "$base/canon/laneS2.log" | cut -c1-200
pgrep -af 'train.py|eval150|infer_test|rule_retune' | cut -c1-90
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; free -g | sed -n 2p
echo "--- OSC (read-only)"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-200
source /etc/profile 2>/dev/null
squeue -u <osc_user> -o "%i %P %T %M %l %R"
squeue -j 8548278 --start -o "%i %S" 2>/dev/null | tail -n 1
sacct -j 8548278 -X -n --format=JobID,State,Start,Elapsed 2>/dev/null
R=/fs/scratch/<osc_project>/bdc
t=$(ls -t $(find $R/code/OpenPCDet/output -path '*pvrcnn_st_osc*' -name 'train_*.log' 2>/dev/null) 2>/dev/null | head -n 1)
[ -n "$t" ] && grep -E 'Train:' "$t" | tail -n 1 | sed -E 's/.*Train: +/Train: /' | cut -c1-150
ls $(find $R/code/OpenPCDet/output -path '*pvrcnn_st_osc*' -name 'checkpoint_epoch_*.pth' 2>/dev/null) 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '
echo; echo "ll files on OSC: $(ls $R/data/claude_twin_ll_v1/points 2>/dev/null | wc -l)"
EOF
~/gcp/globusconnectpersonal -status 2>/dev/null | tr '\n' ' '; echo
