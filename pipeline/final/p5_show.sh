#!/bin/bash
# Show P5 results, job status, and how earlier candidate zips were built.
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
grep -E 'P5_|=== eval|passes|Traceback|Errno|Error' "$base/final/p5_eval_vv.log" | cut -c1-400
echo "--- status $(date -Is)"
bash "$base/canon/status_p.sh" 0 | sed -n 2,9p
tail -n 3 "$base/canon/laneQ.log" | cut -c1-160
echo "--- build scripts"
ls -t "$base/final/"*.sh | head -n 25 | xargs -n1 basename | tr '\n' ' '; echo
ls -t "$base/final/"*.py | xargs -n1 basename | tr '\n' ' '; echo
ls -td "$base/final/"*/ | head -n 12
ls -t "$base/final/"*.zip 2>/dev/null | head -n 5
