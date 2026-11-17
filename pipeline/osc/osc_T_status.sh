#!/bin/bash
# READ-ONLY: state and latest training line of T1 (8563861) and T2 (8563862).
sleep "${1:-0}"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60 <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-210
source /etc/profile 2>/dev/null
R=/fs/scratch/<osc_project>/bdc
squeue -u <osc_user> -o "%i %T %M %l %R"
for t in pvrcnn_truck_T1 second_truck_T2; do
  l=$(ls -t $(find $R/code/OpenPCDet/output -path "*/$t/*" -name 'train_*.log' 2>/dev/null) 2>/dev/null | head -n 1)
  echo "$t: $(grep -E 'Train:' "$l" 2>/dev/null | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-120)"
  grep -E 'Traceback|Error|out of memory' "$l" 2>/dev/null | tail -n 2
done
for j in T1 T2; do o=$(ls -t $R/logs/${j}_*.out 2>/dev/null | head -n 1); [ -n "$o" ] && tail -n 2 "$o"; e=${o%.out}.err; grep -E 'Traceback|Error' "$e" 2>/dev/null | tail -n 2; done
EOF
