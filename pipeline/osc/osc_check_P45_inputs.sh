#!/bin/bash
# READ-ONLY: are the datasets/configs that P4/P5 would use present on OSC?
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60 ${OSC_USER}@ascend.osc.edu "OSC_USER=${OSC_USER} OSC_PROJECT=${OSC_PROJECT} bash -s" 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-200
R=/fs/scratch/${OSC_PROJECT}/bdc
ls $R/cfg/
ls $R/data/ 2>/dev/null || ls $R
grep -nE 'DATA|CFG|data' $R/cfg/pvrcnn_train.sbatch | head -20
source /etc/profile 2>/dev/null; squeue -u ${OSC_USER} -h | wc -l
EOF
