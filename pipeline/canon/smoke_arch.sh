#!/bin/bash
# Local smoke test (RTX 4070, 4 min each): do the new pedestrian Part-A2 / Voxel R-CNN configs build and train (loss finite)?
canon="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
P="$HOME/projects/detector-program/env310/bin/python"
cd "$HOME/projects/detector-program/OpenPCDet/tools"
for c in parta2_ped voxelrcnn_ped; do
  echo "=== $c $(date -Is)"
  timeout 240 "$P" -u train.py --cfg_file "$canon/$c.yaml" --batch_size 2 --epochs 1 --workers 4 --extra_tag smoke \
    --skip_builtin_eval --logger_iter_interval 20 --wo_gpu_stat \
    --set DATA_CONFIG.DATA_PATH "$HOME/projects/bdc2026-track-b/data/claude_canon_all_twin_st_v1" 2>&1 \
    | grep -E -i 'loss|error|traceback|Exception|it/s|s/it|total_it|accumulated' | tail -n 12
  echo "rc ${PIPESTATUS[0]} (124 = stopped by timeout = OK)"
done
echo SMOKE_DONE
