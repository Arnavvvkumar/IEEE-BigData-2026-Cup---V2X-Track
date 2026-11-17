#!/bin/bash
# Copy the env-build and smoke scripts to OSC and start the build on the login node.
# Does not stop upload_rsync.sh.
set -euo pipefail
cd "$(dirname "$0")"
bash ./make_requirements_osc.sh
SSHCMD="ssh -i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 -o ServerAliveInterval=30"
R=/fs/scratch/<osc_project>/bdc
$SSHCMD <osc_user>@ascend.osc.edu "mkdir -p $R/cfg $R/logs $R/tmp $R/.pip-cache" 2>/dev/null
rsync -a -e "$SSHCMD" \
  requirements_osc.txt constraints.txt remote_build_env.sh smoke_check.py smoke_debug.sbatch \
  "<osc_user>@ascend.osc.edu:$R/cfg/"
# nohup plus a short sleep keeps the build alive after this ssh session closes.
$SSHCMD <osc_user>@ascend.osc.edu "nohup bash -lc 'bash $R/cfg/remote_build_env.sh' > $R/logs/build_env.nohup 2>&1 < /dev/null & echo STARTED \$!; sleep 2; tail -n 15 $R/logs/build_env.log 2>/dev/null || echo 'log not created yet'" \
  2>/dev/null | grep -v -E '^\*|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials|^$'
