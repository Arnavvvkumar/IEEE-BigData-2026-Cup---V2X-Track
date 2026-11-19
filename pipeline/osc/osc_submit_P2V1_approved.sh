#!/bin/bash
# Approved by the user on 2026-11-13 ("DO 1+2"): exactly two jobs, each 1 A100, 8 cores, 32 GB, 6 h, nextgen, 5 epochs, batch 4,
# data claude_canon_all_twin_st_v1:  P2 = PV-RCNN pedestrian-only (new seed),  V1 = PV-RCNN vehicle-only.
here="${BDC_ROOT}/experiments/claude_carla_overnight"
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
R=/fs/scratch/<osc_project>/bdc
ssh $K <osc_user>@ascend.osc.edu "cat > $R/cfg/pvrcnn_veh.yaml" < "$here/canon/pvrcnn_veh.yaml"
ssh $K <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E "$F" | cut -c1-200
R=/fs/scratch/<osc_project>/bdc
source /etc/profile 2>/dev/null
grep -A1 CLASS_NAMES $R/cfg/pvrcnn_veh.yaml | head -n 2
for j in "P2 pvrcnn_ped_P2 pvrcnn_ped.yaml" "V1 pvrcnn_veh_V1 pvrcnn_veh.yaml"; do
  set -- $j
  sbatch --account=<osc_project> --partition=nextgen --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gpus-per-node=1 --time=06:00:00 \
    --output=$R/logs/$1_%j.out --error=$R/logs/$1_%j.err \
    --export=ALL,TAG=$2,CFG=$3,DATA=claude_canon_all_twin_st_v1,EPOCHS=5,BS=4,WORKERS=6 $R/cfg/pvrcnn_train.sbatch
done
sleep 8
squeue -u <osc_user> -o "%i %j %T %M %l %C %m %R"
EOF
