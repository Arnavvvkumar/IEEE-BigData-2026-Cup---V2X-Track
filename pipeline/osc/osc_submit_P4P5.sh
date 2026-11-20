#!/bin/bash
# Approved by the user on 2026-11-14 ("Yes, submit all 3"). Uploads one config file, then submits exactly three jobs,
# each 1 A100, 8 cores, 32 GB, 6 h, nextgen, 5 epochs, batch 4 (same as P3):
#   P4 = PV-RCNN pedestrian-only + SE-SSD part dropout/sparsify/swap augmentation (canon/pvrcnn_ped_aug.yaml), data claude_canon_all_twin_st_v1
#   P5 = PV-RCNN pedestrian-only, plain recipe (pvrcnn_ped.yaml), different synthetic data claude_canon_all_v1 (ensemble diversity)
#   P6 = both ideas together: augmentation + claude_canon_all_v1   (three jobs in total)
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
scp $K "${BDC_ROOT}/experiments/claude_carla_overnight/canon/pvrcnn_ped_aug.yaml" ${OSC_USER}@ascend.osc.edu:/fs/scratch/${OSC_PROJECT}/bdc/cfg/ 2>&1 | grep -v -E "$F"
ssh $K ${OSC_USER}@ascend.osc.edu "OSC_USER=${OSC_USER} OSC_PROJECT=${OSC_PROJECT} bash -s" 2>&1 << 'EOF' | grep -v -E "$F" | cut -c1-200
R=/fs/scratch/${OSC_PROJECT}/bdc
source /etc/profile 2>/dev/null
for j in "P4 pvrcnn_ped_P4 pvrcnn_ped_aug.yaml claude_canon_all_twin_st_v1" "P5 pvrcnn_ped_P5 pvrcnn_ped.yaml claude_canon_all_v1" "P6 pvrcnn_ped_P6 pvrcnn_ped_aug.yaml claude_canon_all_v1"; do
  set -- $j
  sbatch --account=${OSC_PROJECT} --partition=nextgen --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gpus-per-node=1 --time=06:00:00 \
    --output=$R/logs/$1_%j.out --error=$R/logs/$1_%j.err \
    --export=ALL,TAG=$2,CFG=$3,DATA=$4,EPOCHS=5,BS=4,WORKERS=6 $R/cfg/pvrcnn_train.sbatch
done
sleep 8
squeue -u ${OSC_USER} -o "%i %j %T %M %l %C %m %R"
EOF
