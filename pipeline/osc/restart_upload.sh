#!/bin/bash
# Inspect, clean up stale rsync processes, restart the resumable data upload, report what is on OSC.
cd "${BDC_ROOT}/experiments/claude_carla_overnight/osc"
echo "stale rsync procs before: $(pgrep -fc 'rsync -aL')"
ps -eo pid,etime,args | grep 'rsync -aL' | grep -v grep | head -n 4 | cut -c1-120
pkill -f upload_rsync.sh; pkill -f 'rsync -aL'; sleep 3
echo "after cleanup: $(pgrep -fc 'rsync -aL')"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=30 <osc_user>@ascend.osc.edu 'du -sm /fs/scratch/<osc_project>/bdc/data/* /fs/scratch/<osc_project>/bdc/code/OpenPCDet 2>&1; ls /fs/scratch/<osc_project>/bdc/code/OpenPCDet | head -n 8' 2>/dev/null | grep -v -E '^\*|^\s*$|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials'
nohup bash upload_rsync.sh > upload_rsync2.log 2>&1 &
sleep 45
echo "restarted: $(pgrep -fc 'upload_rsync.sh') script, $(pgrep -fc 'rsync -aL') rsync"
tail -n 3 upload_rsync2.log | cut -c1-140
