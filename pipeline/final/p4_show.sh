#!/bin/bash
# Show P4 progress, the size-rule grid so far, Lane O standalone result, and job status.
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
grep -E 'P4_|=== eval|passes|Traceback|Errno|Error|INFER' "$base/final/p4_eval_o.log" | cut -c1-400
echo "--- size rule"
grep -E '===|base|DONE|Traceback|Error' "$base/coop/run_size_rule.log"
grep -E '^grow' "$base/coop/run_size_rule.log" | wc -l
echo "--- lane O"
grep -E 'ep14|ep13' "$base/canon/laneO.log" | cut -c1-220
echo "--- status $(date -Is)"
bash "$base/canon/status_p.sh" 0 | sed -n 3,9p
