#!/bin/bash
# Copy the fixed build script to OSC and start it in the background there (does not touch the data upload).
d="${BDC_ROOT}/experiments/claude_carla_overnight/osc"
K=~/.ssh/osc_ed25519
scp -i $K -o BatchMode=yes "$d/remote_build_env.sh" ${OSC_USER}@ascend.osc.edu:/fs/scratch/${OSC_PROJECT}/bdc/cfg/remote_build_env.sh 2>&1 | tail -n 1
ssh -i $K -o BatchMode=yes ${OSC_USER}@ascend.osc.edu 'R=/fs/scratch/${OSC_PROJECT}/bdc; rm -f $R/logs/build_env.pid; nohup bash $R/cfg/remote_build_env.sh > $R/logs/build_env_run2.out 2>&1 & sleep 2; echo started' 2>/dev/null | tail -n 1
