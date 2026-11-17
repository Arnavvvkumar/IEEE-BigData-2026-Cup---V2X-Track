#!/bin/bash
# Lane H: SECOND on the merged set (canonical full set + site twin), mixed precision.
# Waits for the merged dataset, then replaces the third old-data SECOND seed (low value now) with this run.
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
G='-3.15 -3.0 -2.85 -2.7'
cd "$here"
while [ ! -f "$D/claude_canon_all_twin_v1/manifest.json" ]; do sleep 20; done
for p in $(pgrep -f 'bash laneF\.sh'); do kill "$p"; done
sleep 1
for p in $(pgrep -f 'run_arm_par_ns.sh canon_all_rs3_amp'); do kill "$p"; done
for p in $(pgrep -f 'extra_tag canon_all_rs3_amp'); do kill "$p"; done
sleep 25
export WORKERS=3
bash "$here/run_arm_par_ns.sh" second_twin_s1 "$here/second_canon_all_v1.yaml" "$D/claude_canon_all_twin_v1" 12 "$G" --use_amp
echo LANEH_DONE
