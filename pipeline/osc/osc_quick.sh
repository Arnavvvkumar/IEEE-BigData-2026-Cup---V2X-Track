#!/bin/bash
# READ-ONLY: PV-RCNN job progress on OSC, checkpoint copy and combined-scoring status on the laptop.
date +%H:%M
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 ${OSC_USER}@ascend.osc.edu "OSC_USER=${OSC_USER} OSC_PROJECT=${OSC_PROJECT} bash -s" 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-210
source /etc/profile 2>/dev/null
squeue -u ${OSC_USER} -o "%i %T %M %l"
R=/fs/scratch/${OSC_PROJECT}/bdc
t=$(ls -t $(find $R/code/OpenPCDet/output -path '*pvrcnn_st_osc*' -name 'train_*.log' 2>/dev/null) 2>/dev/null | head -n 1)
[ -n "$t" ] && grep -E 'Train:' "$t" | tail -n 1 | sed -E 's/.*Train: +/Train: /' | cut -c1-175
find $R/code/OpenPCDet/output -path '*pvrcnn_st_osc*' -name 'checkpoint_epoch_*.pth' 2>/dev/null | sed -E 's/.*epoch_([0-9]+).pth/\1/' | sort -n | tr '\n' ' '; echo "(checkpoints)"
echo "ll files on OSC: $(ls $R/data/claude_twin_ll_v1/points 2>/dev/null | wc -l)"
EOF
echo "--- laptop"
tail -n 6 "${BDC_ROOT}/experiments/claude_carla_overnight/final/p12_combine.log" | cut -c1-220
