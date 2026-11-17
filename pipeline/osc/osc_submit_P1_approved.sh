#!/bin/bash
# Approved by the user on 2026-11-13 ("Do it, I approve"): P1 = PV-RCNN pedestrian-only, 5 epochs, batch 4,
# 1 A100, 8 cores, 32 GB, 6 h, nextgen, data claude_canon_all_twin_st_v1. Nothing else.
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
R=/fs/scratch/<osc_project>/bdc
ssh $K <osc_user>@ascend.osc.edu "cat > $R/cfg/pvrcnn_ped.yaml" < "$here/canon/pvrcnn_ped.yaml"
ssh $K <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E "$F" | cut -c1-200
R=/fs/scratch/<osc_project>/bdc
source /etc/profile 2>/dev/null
grep -A2 CLASS_NAMES $R/cfg/pvrcnn_ped.yaml | head -n 3
sbatch --account=<osc_project> --partition=nextgen --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gpus-per-node=1 --time=06:00:00 \
  --output=$R/logs/P1_%j.out --error=$R/logs/P1_%j.err \
  --export=ALL,TAG=pvrcnn_ped_P1,CFG=pvrcnn_ped.yaml,DATA=claude_canon_all_twin_st_v1,EPOCHS=5,BS=4,WORKERS=6 $R/cfg/pvrcnn_train.sbatch
sleep 8
squeue -u <osc_user> -o "%i %j %T %M %l %C %m %R"
EOF
