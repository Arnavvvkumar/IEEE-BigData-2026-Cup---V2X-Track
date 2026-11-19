#!/bin/bash
# READ-ONLY: local training status, then what is on OSC after the Globus transfer (data folders, frame counts, queue).
echo "ft4: $(grep -E 'Train:' ${BDC_PROGRAM}/vnx010_pv_ft4.log 2>/dev/null | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-105)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=========== OSC"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-200
R=/fs/scratch/<osc_project>/bdc
for d in $R/data/*/; do
  echo "$(basename $d): $(du -sh $d 2>/dev/null | cut -f1) | train bins $(ls $d/points 2>/dev/null | wc -l) | top $(ls $d | tr '\n' ' ' | cut -c1-90)"
done
ls $R
ls $R/cfg 2>/dev/null | tr '\n' ' '; echo
source /etc/profile 2>/dev/null
squeue -u <osc_user> 2>/dev/null | head -n 5
EOF
