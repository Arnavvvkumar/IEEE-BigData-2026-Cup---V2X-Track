#!/bin/bash
# Approved by the user on 2026-11-14 ("Yes, submit both"): exactly two jobs.
# Uploads one config file, then submits exactly two jobs, each 1 A100, 8 cores, 32 GB, 6 h, nextgen, 5 epochs, batch 4 (same as P1-P6):
#   VR1 = Voxel R-CNN pedestrian-only (new architecture), data claude_canon_all_twin_st_v1 (same data as P1)
#   VR2 = Voxel R-CNN pedestrian-only, data claude_canon_all_v1 (different synthetic data, for diversity)
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
scp $K "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon/voxelrcnn_ped.yaml" <osc_user>@ascend.osc.edu:/fs/scratch/<osc_project>/bdc/cfg/ 2>&1 | grep -v -E "$F"
ssh $K <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E "$F" | cut -c1-200
R=/fs/scratch/<osc_project>/bdc
source /etc/profile 2>/dev/null
for j in "VR1 vrcnn_ped_VR1 voxelrcnn_ped.yaml claude_canon_all_twin_st_v1" "VR2 vrcnn_ped_VR2 voxelrcnn_ped.yaml claude_canon_all_v1"; do
  set -- $j
  sbatch --account=<osc_project> --partition=nextgen --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gpus-per-node=1 --time=06:00:00 \
    --output=$R/logs/$1_%j.out --error=$R/logs/$1_%j.err \
    --export=ALL,TAG=$2,CFG=$3,DATA=$4,EPOCHS=5,BS=4,WORKERS=6 $R/cfg/pvrcnn_train.sbatch
done
sleep 8
squeue -u <osc_user> -o "%i %j %T %M %l %C %m %R"
EOF
