#!/bin/bash
# READ-ONLY: is the long-low truck set complete on OSC? Compares file counts with the laptop copy. Also GPU memory of job 1.
D="${BDC_DATA}/claude_twin_ll_v1"
echo "laptop: points $(ls $D/points | wc -l) labels $(ls $D/labels | wc -l) | $(ls $D | tr '\n' ' ')"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60 <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-200
R=/fs/scratch/<osc_project>/bdc
D=$R/data/claude_twin_ll_v1
echo "OSC:    points $(ls $D/points 2>/dev/null | wc -l) labels $(ls $D/labels 2>/dev/null | wc -l) | $(ls $D 2>/dev/null | tr '\n' ' ')"
source /etc/profile 2>/dev/null
squeue -u <osc_user>
sacct -j 8548278 -X -n --format=JobID,Elapsed,MaxRSS,TRESUsageInMax%60 2>/dev/null | head -n 3
grep -h -E 'Data time|Batch time' $(ls -t $(find $R/code/OpenPCDet/output -path '*pvrcnn_st_osc/*' -name 'train_*.log') | head -n 1) 2>/dev/null | tail -n 1 | grep -oE 'Data time: [0-9.()]+ +Forward time: [0-9.()]+ +Batch time: [0-9.()]+'
EOF
