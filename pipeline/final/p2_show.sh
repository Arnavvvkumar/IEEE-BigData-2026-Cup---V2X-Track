#!/bin/bash
# Show the P2 (small-truck SECOND) ensemble results plus overnight job status.
base="${BDC_ROOT}/experiments/claude_carla_overnight"
grep -E 'P2_|=== eval|passes|Traceback|Errno|Error' "$base/final/p2_eval_st.log" | cut -c1-400
echo "--- status $(date -Is)"
bash "$base/canon/status_all.sh"
