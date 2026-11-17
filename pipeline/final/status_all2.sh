#!/bin/bash
# Read-only: what is running or waiting right now (laptop, OSC, Codex).
date +%H:%M
echo "--- laptop processes"
for p in p22_P2V1 eval150 infer_test 'codex_final' realism_exp train.py; do echo "$p: $(pgrep -fc "$p")"; done
echo "--- P22 (waits for OSC P2 / V1) tail"; tail -n 3 "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final/p22_P2V1.log" | cut -c1-200
echo "--- GPU / memory"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; free -g | sed -n 2p
echo "--- OSC (read-only)"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60 <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-200
source /etc/profile 2>/dev/null
R=/fs/scratch/<osc_project>/bdc
squeue -u <osc_user> -o "%i %j %T %M %l %R"
for t in pvrcnn_ped_P2 pvrcnn_veh_V1; do
  l=$(ls -t $(find $R/code/OpenPCDet/output -path "*/$t/*" -name 'train_*.log' 2>/dev/null) 2>/dev/null | head -n 1)
  echo "$t: $(grep -E 'Train:' "$l" 2>/dev/null | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-110)"
done
EOF
