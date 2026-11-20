#!/bin/bash
# Re-send the truncated files listed in bad_files.txt (synthetic training frames only), verify them, then submit the full PV-RCNN job.
here="${BDC_ROOT}/experiments/claude_carla_overnight/osc"
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
cd ${BDC_DATA} || exit 1
rsync -aL --files-from="$here/bad_files.txt" -e "ssh -i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40" . ${OSC_USER}@ascend.osc.edu:/fs/scratch/${OSC_PROJECT}/bdc/data/ 2>&1 | grep -v -E "$F"
echo "rsync rc ${PIPESTATUS[0]}"
for f in $(cat "$here/bad_files.txt"); do echo "$(stat -L -c %s "$f") $f"; done > /tmp/bad_sizes.txt
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 ${OSC_USER}@ascend.osc.edu "OSC_USER=${OSC_USER} OSC_PROJECT=${OSC_PROJECT} bash -s" 2>&1 < <(
  echo 'R=/fs/scratch/${OSC_PROJECT}/bdc; source /etc/profile 2>/dev/null; ok=1'
  while read -r size f; do echo "[ \"\$(stat -c %s \$R/data/$f)\" = \"$size\" ] || { echo SIZE_MISMATCH $f; ok=0; }"; done < /tmp/bad_sizes.txt
  cat << 'EOF'
if [ $ok = 1 ]; then
  echo "all re-sent files match local sizes"
  sbatch --account=${OSC_PROJECT} --partition=nextgen --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=64G --gpus-per-node=1 --time=16:00:00 \
    --output=$R/logs/pvfull_%j.out --error=$R/logs/pvfull_%j.err \
    --export=ALL,TAG=pvrcnn_st_osc,EPOCHS=12,BS=4,WORKERS=8 $R/cfg/pvrcnn_train.sbatch
fi
squeue -u ${OSC_USER}
EOF
) | grep -v -E "$F" | cut -c1-200
