#!/bin/bash
# Local (RTX 4070) training: pedestrian-only Voxel R-CNN (new architecture for ensemble diversity), synthetic data only, from scratch.
# Same data as P1 (claude_canon_all_twin_st_v1), 5 epochs, batch 2 (smoke test: ~49 min/epoch).
canon="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
P="${BDC_PYTHON}"
cd "${BDC_PROGRAM}/OpenPCDet/tools"
echo "START $(date -Is)"
"$P" -u train.py --cfg_file "$canon/voxelrcnn_ped.yaml" --batch_size 2 --epochs 5 --workers 4 --extra_tag vrcnn_ped_L1 \
  --skip_builtin_eval --logger_iter_interval 200 --wo_gpu_stat --max_ckpt_save_num 5 \
  --set DATA_CONFIG.DATA_PATH "${BDC_DATA}/claude_canon_all_twin_st_v1" 2>&1 | grep -E -i 'Train: .*00%\]|error|traceback|epoch|save'
echo "TRAIN_END $(date -Is) rc ${PIPESTATUS[0]}"
