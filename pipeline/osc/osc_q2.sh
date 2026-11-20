#!/bin/bash
# READ-ONLY: my OSC queue + latest training lines of P2 and V1.
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60 ${OSC_USER}@ascend.osc.edu "OSC_USER=${OSC_USER} OSC_PROJECT=${OSC_PROJECT} bash -s" 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-200
source /etc/profile 2>/dev/null
R=/fs/scratch/${OSC_PROJECT}/bdc
squeue -u ${OSC_USER} -o "%i %j %T %M %l %R"
for t in pvrcnn_ped_P2 pvrcnn_veh_V1; do
  l=$(ls -t $(find $R/code/OpenPCDet/output -path "*/$t/*" -name 'train_*.log' 2>/dev/null) 2>/dev/null | head -n 1)
  echo "$t: $(grep -E 'Train:' "$l" 2>/dev/null | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-110)"
done
EOF
date +%H:%M
