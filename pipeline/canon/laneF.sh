#!/bin/bash
# Lane F: when lane C2 moves on to PV-RCNN (estimated 25 h for 8 epochs; PV-RCNN was the weakest levelled model earlier),
# stop that step and use the GPU lane for a third SECOND full-set seed with mixed precision (--use_amp) instead.
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
while ! pgrep -f 'extra_tag pvrcnn_all_s1234' > /dev/null; do sleep 30; done
for p in $(pgrep -f 'bash laneC2\.sh'); do kill "$p"; done
sleep 1
for p in $(pgrep -f 'run_arm_par.sh pvrcnn_all_s1234'); do kill "$p"; done
for p in $(pgrep -f 'extra_tag pvrcnn_all_s1234'); do kill "$p"; done
sleep 20
echo "PV-RCNN step stopped at $(date)"
export WORKERS=3
bash "$here/run_arm_par_ns.sh" canon_all_rs3_amp "$here/second_canon_all_v1.yaml" "$D/claude_canon_all_v1" 12 "$G" --use_amp
echo LANEF_DONE
