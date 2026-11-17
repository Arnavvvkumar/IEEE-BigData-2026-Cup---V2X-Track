#!/bin/bash
# Show P3 (size-aware pedestrian model) results plus job status.
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
grep -E 'P3_|=== eval|passes|^vnx| n |Traceback|Errno|Error|INFER' "$base/final/p3_eval_pv.log" | cut -c1-400
echo "--- status $(date -Is)"
bash "$base/canon/status_p.sh" 0 | sed -n 2,9p
tail -n 3 "$base/canon/laneO.log" | cut -c1-220
