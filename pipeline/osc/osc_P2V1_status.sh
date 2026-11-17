#!/bin/bash
# READ-ONLY: state and latest training line of P2 (8571774) and V1 (8571775).
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60 <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-210
source /etc/profile 2>/dev/null
R=/fs/scratch/<osc_project>/bdc
squeue -u <osc_user> -o "%i %j %T %M %l %R"
for t in pvrcnn_ped_P2 pvrcnn_veh_V1; do
  l=$(ls -t $(find $R/code/OpenPCDet/output -path "*/$t/*" -name 'train_*.log' 2>/dev/null) 2>/dev/null | head -n 1)
  echo "$t: $(grep -E 'Train:' "$l" 2>/dev/null | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-150)"
done
date
EOF
