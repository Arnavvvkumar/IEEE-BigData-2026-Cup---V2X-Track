#!/bin/bash
# READ-ONLY: queue plus the tail of the newest PV-RCNN job logs and training log on OSC.  usage: osc_job_check.sh [sleep secs] [log prefix]
sleep "${1:-0}"
P="${2:-pv}"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 <osc_user>@ascend.osc.edu "P=$P bash -s" 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-230
R=/fs/scratch/<osc_project>/bdc
source /etc/profile 2>/dev/null
squeue -u <osc_user>
o=$(ls -t $R/logs/${P}*.out 2>/dev/null | head -n 1); e=${o%.out}.err
echo "--- $o"; grep -v -E 'it/s\]|s/it\]' "$o" 2>/dev/null | tail -n 6
echo "--- err"; grep -v -E 'Warning|warn|it/s\]|s/it\]|^\s*$' "$e" 2>/dev/null | tail -n 8
t=$(ls -t $(find $R/code/OpenPCDet/output -name 'train_*.log' 2>/dev/null) 2>/dev/null | head -n 1)
echo "--- $t"; grep -E 'Train:|Error|Traceback|out of memory' "$t" 2>/dev/null | tail -n 3 | sed -E 's/.*Train: +/Train: /'
EOF
