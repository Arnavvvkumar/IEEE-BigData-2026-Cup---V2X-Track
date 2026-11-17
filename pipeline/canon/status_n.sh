#!/bin/bash
# Lane N result (dev50 standalone) and P3 progress.
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
tail -n 8 laneN.log | cut -c1-220
echo "--- p3"
tail -n 6 ../final/p3_eval_pv.log | cut -c1-300
bash status_p.sh 0 | sed -n 2,9p
