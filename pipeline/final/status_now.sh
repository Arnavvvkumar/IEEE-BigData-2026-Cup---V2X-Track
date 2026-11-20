#!/bin/bash
# Read-only status: OSC jobs, laptop scoring runs, Codex, GPU, memory.
b="${BDC_ROOT}/experiments"
date +%H:%M
echo "--- laptop processes"
for p in p19_T1P1 p20_T1ep2 eval150 infer_test 'codex_final' realism; do echo "$p: $(pgrep -fc "$p")"; done
echo "--- T1 epoch-2 scoring (p20)"; grep -E 'T1 epoch 2|P20_DONE|NO_T1|Traceback' "$b/claude_carla_overnight/final/p20_T1ep2.log" | cut -c1-215
echo "--- P19 (waits for P1)"; tail -n 2 "$b/claude_carla_overnight/final/p19_T1P1.log" | cut -c1-200
echo "--- Codex ped TTA"; grep -E 'pass |v9|ped_tta|score' "$b/codex_final/ped_tta.log" | tail -n 3 | cut -c1-180
echo "--- GPU / memory"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; free -g | sed -n 2p
echo "--- OSC (read-only)"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60 ${OSC_USER}@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-200
source /etc/profile 2>/dev/null
R=/fs/scratch/${OSC_PROJECT}/bdc
squeue -u ${OSC_USER} -o "%i %j %T %M %l"
l=$(ls -t $(find $R/code/OpenPCDet/output -path '*/pvrcnn_ped_P1/*' -name 'train_*.log' 2>/dev/null) 2>/dev/null | head -n 1)
echo "P1: $(grep -E 'Train:' "$l" 2>/dev/null | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-120)"
EOF
