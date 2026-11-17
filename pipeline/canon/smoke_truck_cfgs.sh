#!/bin/bash
# 75-second laptop smoke test of the truck-only configs: does the model build, does the loss compute, do a few steps run?
repo="$HOME/projects/detector-program/OpenPCDet"; P="$HOME/projects/detector-program/env310/bin/python"
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data/claude_canon_all_twin_ll_v1"
cd "$repo/tools"
for c in second_truck pvrcnn_truck; do
  bs=2; [ $c = pvrcnn_truck ] && bs=1
  timeout 75 "$P" -u train.py --cfg_file "$here/$c.yaml" --batch_size $bs --epochs 1 --workers 2 --extra_tag smoke_$c \
    --skip_builtin_eval --logger_iter_interval 10 --wo_gpu_stat --max_ckpt_save_num 1 --set DATA_CONFIG.DATA_PATH "$D" > /tmp/smoke_$c.log 2>&1
  echo "== $c rc $? (124 = stopped by the time limit, expected)"
  grep -E 'Train:|Traceback|Error|error' /tmp/smoke_$c.log | tail -n 3 | sed -E 's/.*Train: +/Train: /' | cut -c1-170
done
