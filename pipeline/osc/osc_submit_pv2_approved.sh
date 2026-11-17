#!/bin/bash
# Submits OSC option A EXACTLY as the user approved on 2026-11-13 ("put in req for Option A"):
# second PV-RCNN seed, 1 A100, 8 cores, 32 GB, 5 h limit, nextgen, 4 epochs, batch 4. Nothing else.
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60 <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E "$F" | cut -c1-200
source /etc/profile 2>/dev/null
echo "queue before:"; squeue -u <osc_user> -o "%i %P %j %T %M %l"
sbatch --account=<osc_project> --partition=nextgen --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G \
  --gpus-per-node=1 --time=05:00:00 \
  --output=/fs/scratch/<osc_project>/bdc/logs/pvfull2_%j.out --error=/fs/scratch/<osc_project>/bdc/logs/pvfull2_%j.err \
  --export=ALL,TAG=pvrcnn_st_osc_s2,EPOCHS=4,BS=4,WORKERS=6 /fs/scratch/<osc_project>/bdc/cfg/pvrcnn_train.sbatch
sleep 5
echo "queue after:"; squeue -u <osc_user> -o "%i %P %j %T %M %l %C %m %R"
EOF
