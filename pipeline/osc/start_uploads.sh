#!/bin/bash
# Start both dataset uploads in the background and report the remote size after a minute.
cd "${BDC_ROOT}/experiments/claude_carla_overnight/osc"
D="${BDC_DATA}"
echo "local files: canon $(find -L $D/claude_canon_all_v1 -type f | wc -l)  twin $(find -L $D/claude_twin_v1 -type f | wc -l)"
nohup bash upload_data.sh claude_canon_all_v1 > upload_canon.log 2>&1 &
sleep 2
nohup bash upload_data.sh claude_twin_v1 > upload_twin.log 2>&1 &
sleep 60
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes <osc_user>@ascend.osc.edu 'du -sm /fs/scratch/<osc_project>/bdc/data/* 2>/dev/null' 2>/dev/null | tail -n 3
