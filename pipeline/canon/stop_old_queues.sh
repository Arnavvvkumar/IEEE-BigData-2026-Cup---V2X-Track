#!/bin/bash
# Stop the sequential queues 3 and 4 (their current training child keeps running and still gets scored by its run_arm.sh).
for p in $(pgrep -f 'bash queue[34]\.sh'); do kill "$p"; done
sleep 1
pgrep -af 'queue|lane|run_arm' | cut -c1-100
