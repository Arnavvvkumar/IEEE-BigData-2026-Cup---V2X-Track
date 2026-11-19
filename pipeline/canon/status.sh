#!/bin/bash
# Read-only status of the training lanes.
cd "${BDC_PROGRAM}"
for f in "$@"; do
  echo "== $f"
  tail -c 3000 "$f.log" | tr '\r' '\n' | grep -E 'epochs:|Traceback|Error|Killed' | tail -n 2 | cut -c1-160
done
pgrep -af 'extra_tag' | sed 's/.*extra_tag \([a-z0-9_]*\).*/\1/' | sort | uniq -c
free -m | sed -n 2p
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
