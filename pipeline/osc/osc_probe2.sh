#!/bin/bash
# READ-ONLY probe of OSC: partitions, the GPU flag the smoke job used, cfg/code/log folders, python env.
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-220
R=/fs/scratch/<osc_project>/bdc
source /etc/profile 2>/dev/null
echo "--- partitions"; sinfo -s 2>/dev/null | head -n 20
echo "--- smoke logs"; ls -t $R/logs | head -n 12
grep -h -E 'GPU_ARGS|SMOKE_|device|Submitted' $R/logs/* 2>/dev/null | sort -u | tail -n 12
echo "--- cfg"; ls $R/cfg
echo "--- code"; ls $R/code; ls $R/code/OpenPCDet/tools | head -n 20
echo "--- out"; ls $R/out | head
echo "--- val split of canon"; wc -l $R/data/claude_canon_all_v1/ImageSets/*.txt
ls $R/data/claude_twin_v1/ImageSets $R/data/claude_twin_st_v1/ImageSets
echo "--- quota"; df -h /fs/scratch/<osc_project> | tail -n 1
EOF
