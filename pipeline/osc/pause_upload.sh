#!/bin/bash
# Temporarily PAUSE (SIGSTOP) the OSC upload rsync processes so the Globus login can get bandwidth; auto-resume (SIGCONT) after $1 seconds (default 360).
secs=${1:-360}
pids=$(pgrep -f 'rsync -aL' | tr '\n' ' ')
echo "pausing: $pids"
kill -STOP $pids 2>/dev/null
( sleep "$secs"; kill -CONT $pids 2>/dev/null; echo "resumed $(date -Is)" >> /tmp/pause_upload.log ) > /dev/null 2>&1 &
disown
sleep 3
timeout 20 curl -s -o /dev/null -w "auth.globus.org handshake test: HTTP %{http_code} in %{time_total}s\n" https://auth.globus.org/ || echo "auth.globus.org still unreachable"
