#!/bin/bash
# Re-run the (fixed) code push in the background and report upload state. Safe to repeat.
cd "${BDC_ROOT}/experiments/claude_carla_overnight/osc"
echo "uploader procs: $(pgrep -fc 'upload_rsync.sh')  rsync procs: $(pgrep -fc 'rsync -aL')"
tail -n 3 upload_rsync.log | cut -c1-150
nohup bash push_code.sh > push_code2.log 2>&1 &
sleep 90
grep -v -E '^\*|^\s*$|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials' push_code2.log | cut -c1-160 | tail -n 15
echo "push still running: $(pgrep -fc 'push_code.sh')"
