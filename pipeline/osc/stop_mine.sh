#!/bin/bash
# Stop ONLY the uploader this agent started (upload_rsync.sh and its '--inplace' rsync). Leave Cursor's '--files-from' rsync alone.
pkill -f upload_rsync.sh
pkill -f 'rsync -aL --partial --inplace'
sleep 3
echo "my uploader script: $(pgrep -fc 'upload_rsync.sh')  my rsync (--inplace): $(pgrep -fc 'rsync -aL --partial --inplace')"
echo "Cursor-style rsync (--files-from): $(pgrep -fc 'rsync -aL --partial --size-only --files-from')"
ps -eo pid,etime,args | grep -- '--files-from' | grep -v grep | head -n 3 | cut -c1-140
ls ${BDC_ROOT}/bdc2026-track-b/ | grep -i upload
