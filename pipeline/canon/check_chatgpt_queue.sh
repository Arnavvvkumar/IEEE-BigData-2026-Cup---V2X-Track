#!/bin/bash
# Read-only: what did the ChatGPT-launched queue start, what does it wait for, and what is our own SECOND config?
Q="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/detector_program/second_speed_nov12"
echo "--- queue dir"; ls -la "$Q" 2>&1 | head -n 20
echo "--- queue.log (tail)"; tail -n 15 "$Q/queue.log" 2>&1 | cut -c1-220
echo "--- processes that are not ours"
ps -eo pid,etime,cmd | grep -E 'second_speed|queue|benchmark|run_second' | grep -v grep | cut -c1-200
echo "--- what the queue script would launch"
for f in "$Q"/*.sh "$Q"/*.py; do [ -f "$f" ] && { echo "## $f"; grep -n -E 'wait|pgrep|train.py|--batch_size|--epochs|extra_tag|--cfg_file|DATA_PATH|WORKERS|OMP|nohup|sleep' "$f" | head -n 24 | cut -c1-200; }; done
echo "--- our config (second_canon_all_v1.yaml)"
c="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon/second_canon_all_v1.yaml"
grep -n -A4 'VOXEL_SIZE' "$c" | head -n 6; grep -n -E 'LR:|BATCH_SIZE_PER_GPU|NUM_EPOCHS|MAX_NUMBER_OF_VOXELS' "$c"; grep -n -A3 'MAX_NUMBER_OF_VOXELS' "$c" | head -n 4
echo "--- right now: $(date +%H:%M) | $(free -g | sed -n 2p)"
grep -E 'Train:' ~/projects/detector-program/second_twin_ll_s1.log | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-120
grep -E 'Train:' ~/projects/detector-program/second_twin_ll_s2.log 2>/dev/null | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-120
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
