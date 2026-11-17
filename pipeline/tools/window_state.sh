#!/bin/bash
# Prints READY when the GPU-window conditions in event claude_carla_gpu_window_plan hold; otherwise prints why not.
W="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
Q="$W/experiments/detector_program/overnight_accuracy_queue.log"
reasons=()
grep -q "Waiting for Claude's audited mixed dataset" "$Q" 2>/dev/null || reasons+=("runner_not_waiting")
pgrep -af '[p]ython.*(train\.py|evaluate_openpcdet\.py|test\.py)' >/dev/null && reasons+=("openpcdet_running")
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
[ "$apps" -gt 0 ] && reasons+=("cuda_compute_apps=$apps")
[ "$(date -u +%s)" -lt "$(date -u -d '2026-11-09T04:45:00Z' +%s)" ] && reasons+=("before_04:45Z")
grep -qE '"event": ?"claude_carla_gpu_window_denied"' "$W/coordination/events.jsonl" && reasons+=("DENIED")
if [ ${#reasons[@]} -eq 0 ]; then echo READY; else echo "WAIT ${reasons[*]}"; fi
