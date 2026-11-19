#!/bin/bash
# Copy the fixed build script to OSC and start it in the background there (does not touch the data upload).
d="${BDC_ROOT}/experiments/claude_carla_overnight/osc"
K=~/.ssh/osc_ed25519
scp -i $K -o BatchMode=yes "$d/remote_build_env.sh" <osc_user>@ascend.osc.edu:/fs/scratch/<osc_project>/bdc/cfg/remote_build_env.sh 2>&1 | tail -n 1
ssh -i $K -o BatchMode=yes <osc_user>@ascend.osc.edu 'R=/fs/scratch/<osc_project>/bdc; rm -f $R/logs/build_env.pid; nohup bash $R/cfg/remote_build_env.sh > $R/logs/build_env_run2.out 2>&1 & sleep 2; echo started' 2>/dev/null | tail -n 1
