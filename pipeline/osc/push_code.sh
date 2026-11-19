#!/bin/bash
# Freeze the laptop's detector environment and push OpenPCDet source + our configs/scripts to OSC scratch (small files only).
set -uo pipefail
here="${BDC_ROOT}/experiments/claude_carla_overnight"
cd "$here/osc"
${BDC_PROGRAM}/env310/bin/pip freeze > requirements_laptop.txt 2>/dev/null
wc -l requirements_laptop.txt
SSHCMD="ssh -i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ServerAliveInterval=30"
R=/fs/scratch/<osc_project>/bdc
$SSHCMD <osc_user>@ascend.osc.edu "mkdir -p $R/code $R/cfg $R/out $R/logs" 2>/dev/null
rsync -a --delete -e "$SSHCMD" --exclude output --exclude '__pycache__' --exclude '*.so' --exclude build --exclude '*.egg-info' --exclude .git \
  ${BDC_PROGRAM}/OpenPCDet/ <osc_user>@ascend.osc.edu:$R/code/OpenPCDet/ 2>&1 | tail -n 2
rsync -a -e "$SSHCMD" "$here/canon/"*.yaml "$here/canon/run_arm_par_ns.sh" "$here/twin/merge_sets.py" requirements_laptop.txt <osc_user>@ascend.osc.edu:$R/cfg/ 2>&1 | tail -n 2
$SSHCMD <osc_user>@ascend.osc.edu "ls $R/cfg; du -sh $R/code" 2>/dev/null
