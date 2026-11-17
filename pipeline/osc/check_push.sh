#!/bin/bash
# What actually landed on OSC from push_code.sh (read-only).
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=30 <osc_user>@ascend.osc.edu 'R=/fs/scratch/<osc_project>/bdc; echo "code:"; ls $R/code/OpenPCDet 2>&1 | head -n 12; echo "cfg:"; ls $R/cfg 2>&1 | head -n 30; du -sh $R/code 2>&1; du -sm $R/data/* 2>&1' 2>/dev/null | grep -v -E '^\*|^\s*$|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials'
echo "uploader alive: $(pgrep -fc 'upload_rsync.sh')"
tail -n 2 "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc/upload_rsync.log" | cut -c1-150
