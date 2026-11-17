#!/bin/bash
# Read-only: queue state and last log lines of VR1/VR2 (jobs 8594088, 8594089).
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
ssh $K <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E "$F" | cut -c1-220
R=/fs/scratch/<osc_project>/bdc
squeue -u <osc_user> -o "%i %j %T %M %l %R" 2>/dev/null
squeue -u <osc_user> --start -o "%i %S" 2>/dev/null | tail -n +2
for j in VR1_8594088 VR2_8594089; do
  echo "== $j"; [ -f $R/logs/$j.out ] && tail -n 3 $R/logs/$j.out
  [ -f $R/logs/$j.err ] && grep -E -i 'error|traceback|Train: .*\]' $R/logs/$j.err | tail -n 3
done
ls -la $R/code/OpenPCDet/output/fs/scratch/<osc_project>/bdc/cfg/voxelrcnn_ped/*/ckpt/ 2>/dev/null | tail -n 8
EOF
