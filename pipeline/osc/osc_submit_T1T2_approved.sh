#!/bin/bash
# Approved by the user on 2026-11-13 ("approve T1+T2"). File preparation on OSC, then exactly two jobs:
#   T1: PV-RCNN truck-only, 5 epochs, batch 4, 1 A100, 8 cores, 32 GB, 7 h, nextgen
#   T2: SECOND truck-only, 12 epochs, batch 8, 1 A100, 8 cores, 32 GB, 4 h, nextgen
set -uo pipefail
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
S="ssh $K <osc_user>@ascend.osc.edu"
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
R=/fs/scratch/<osc_project>/bdc
$S "cat > $R/cfg/pvrcnn_train.sbatch" < "$here/osc/pvrcnn_train.sbatch"
$S "cat > $R/cfg/pvrcnn_truck.yaml" < "$here/canon/pvrcnn_truck.yaml"
$S "cat > $R/cfg/second_truck.yaml" < "$here/canon/second_truck.yaml"
$S "cat > $R/cfg/osc_make_llp.py" < "$here/osc/osc_make_llp.py"
$S 'bash -s' 2>&1 << 'EOF' | grep -v -E "$F" | cut -c1-200
R=/fs/scratch/<osc_project>/bdc
source /etc/profile 2>/dev/null
module load python/3.10 2>/dev/null
source $R/env/bin/activate
sed -i 's/\r$//' $R/cfg/pvrcnn_train.sbatch
[ -d $R/data/claude_twin_ll_v1p ] || python $R/cfg/osc_make_llp.py
M=$R/data/claude_canon_all_twin_ll_v1p
[ -f $M/manifest.json ] || python $R/cfg/merge_sets.py $M $R/data/claude_canon_all_v1 $R/data/claude_twin_v1 $R/data/claude_twin_st_v1 $R/data/claude_twin_ll_v1p | grep -E '"train"|"val"'
echo "merged train frames: $(wc -l < $M/ImageSets/train.txt) | broken links: $(find $M/points -xtype l | wc -l)"
if [ "$(find $M/points -xtype l | wc -l)" != 0 ]; then echo "STOP: broken links, nothing submitted"; exit 1; fi
sbatch --account=<osc_project> --partition=nextgen --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gpus-per-node=1 --time=07:00:00 \
  --output=$R/logs/T1_%j.out --error=$R/logs/T1_%j.err \
  --export=ALL,TAG=pvrcnn_truck_T1,CFG=pvrcnn_truck.yaml,DATA=claude_canon_all_twin_ll_v1p,EPOCHS=5,BS=4,WORKERS=6 $R/cfg/pvrcnn_train.sbatch
sbatch --account=<osc_project> --partition=nextgen --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gpus-per-node=1 --time=04:00:00 \
  --output=$R/logs/T2_%j.out --error=$R/logs/T2_%j.err \
  --export=ALL,TAG=second_truck_T2,CFG=second_truck.yaml,DATA=claude_canon_all_twin_ll_v1p,EPOCHS=12,BS=8,WORKERS=6 $R/cfg/pvrcnn_train.sbatch
sleep 8
squeue -u <osc_user> -o "%i %j %T %M %l %C %m %R"
EOF
