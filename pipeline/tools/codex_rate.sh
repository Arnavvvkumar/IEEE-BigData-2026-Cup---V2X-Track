#!/bin/bash
# Read-only: report Codex's active OpenPCDet training iteration rate (from its own log) and GPU memory.
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
LOG=${LOG:-$(ls -t ${BDC_PROGRAM}/*overnight*.log 2>/dev/null | head -1)}
echo "log: $LOG"
grep -E "Train:|it/s|s/it|time_cost" "$LOG" | tail -${1:-4} | cut -c1-260
