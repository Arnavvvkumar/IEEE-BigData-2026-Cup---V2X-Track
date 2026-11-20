#!/bin/bash
# OSC: copy the PV-RCNN job script, build the merged training folder (symlinks) if missing, submit a 12-minute test job.
here="${BDC_ROOT}/experiments/claude_carla_overnight/osc"
S="ssh -i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 ${OSC_USER}@ascend.osc.edu"
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
$S 'cat > /fs/scratch/${OSC_PROJECT}/bdc/cfg/pvrcnn_train.sbatch' < "$here/pvrcnn_train.sbatch" 2>&1 | grep -v -E "$F"
$S "OSC_USER=${OSC_USER} OSC_PROJECT=${OSC_PROJECT} bash -s" 2>&1 << 'EOF' | grep -v -E "$F" | cut -c1-220
R=/fs/scratch/${OSC_PROJECT}/bdc
source /etc/profile 2>/dev/null
module load python/3.10 2>/dev/null
source $R/env/bin/activate
M=$R/data/claude_canon_all_twin_st_v1
if [ ! -f $M/manifest.json ]; then
  rm -rf $M
  python $R/cfg/merge_sets.py $M $R/data/claude_canon_all_v1 $R/data/claude_twin_v1 $R/data/claude_twin_st_v1 | tail -n 8
fi
echo "merged: train $(wc -l < $M/ImageSets/train.txt) val $(wc -l < $M/ImageSets/val.txt) points $(ls $M/points | wc -l)"
sed -i 's/\r$//' $R/cfg/pvrcnn_train.sbatch
sbatch --account=${OSC_PROJECT} --partition=debug-nextgen --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=64G --gpus-per-node=1 --time=00:20:00 \
  --output=$R/logs/pvtest_%j.out --error=$R/logs/pvtest_%j.err \
  --export=ALL,TAG=pvrcnn_test,EPOCHS=1,BS=4,WORKERS=8,LIMIT=720 $R/cfg/pvrcnn_train.sbatch
squeue -u ${OSC_USER}
EOF
