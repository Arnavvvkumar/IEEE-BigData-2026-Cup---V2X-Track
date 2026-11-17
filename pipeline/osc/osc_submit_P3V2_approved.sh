#!/bin/bash
# Approved by the user on 2026-11-14 ("Yes sumbit P3+V2"): exactly two jobs, each 1 A100, 8 cores, 32 GB, 6 h, nextgen, 5 epochs, batch 4,
# data claude_canon_all_twin_st_v1:  P3 = PV-RCNN pedestrian-only (another run),  V2 = PV-RCNN vehicle-only (another run).
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
ssh $K <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E "$F" | cut -c1-200
R=/fs/scratch/<osc_project>/bdc
source /etc/profile 2>/dev/null
for j in "P3 pvrcnn_ped_P3 pvrcnn_ped.yaml" "V2 pvrcnn_veh_V2 pvrcnn_veh.yaml"; do
  set -- $j
  sbatch --account=<osc_project> --partition=nextgen --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gpus-per-node=1 --time=06:00:00 \
    --output=$R/logs/$1_%j.out --error=$R/logs/$1_%j.err \
    --export=ALL,TAG=$2,CFG=$3,DATA=claude_canon_all_twin_st_v1,EPOCHS=5,BS=4,WORKERS=6 $R/cfg/pvrcnn_train.sbatch
done
sleep 8
squeue -u <osc_user> -o "%i %j %T %M %l %C %m %R"
EOF
