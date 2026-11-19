#!/bin/bash
cd "${BDC_ROOT}/experiments/claude_carla_overnight/osc"
echo "STARTED $(date -Is)" > upload_six.log
setsid nohup bash ./upload_six.sh </dev/null >/dev/null 2>&1 &
sleep 20
pgrep -fc 'rsync -aL' ; tail -n 3 upload_six.log
