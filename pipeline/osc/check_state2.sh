#!/bin/bash
# READ-ONLY: build script tail, OSC data sizes, gcc modules, queue
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc"
sed -n 60,140p "$base/remote_build_env.sh"
echo "=========== OSC"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=30 <osc_user>@ascend.osc.edu 'bash -s' 2>/dev/null << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-200
R=/fs/scratch/<osc_project>/bdc
du -sh $R/data/* 2>&1
source /etc/profile 2>/dev/null
module spider gcc 2>&1 | grep -E 'gcc/' | head -n 6
squeue -u <osc_user> | head -n 5
echo "--- compile error lines"
grep -n -E 'icpc|icc|CC=|CXX=|_castu32|error:' $R/logs/build_env.log | head -n 8
EOF
echo "=========== local rsync procs"
pgrep -fc 'rsync' || true
