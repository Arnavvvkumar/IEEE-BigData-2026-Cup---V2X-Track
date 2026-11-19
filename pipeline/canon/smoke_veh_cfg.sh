#!/bin/bash
# 75-second laptop smoke test of the vehicle-only PV-RCNN config.
repo="${BDC_PROGRAM}/OpenPCDet"; P="${BDC_PYTHON}"
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
cd "$here" && "$P" make_truck_cfgs.py | tail -n 1
cd "$repo/tools"
timeout 75 "$P" -u train.py --cfg_file "$here/pvrcnn_veh.yaml" --batch_size 1 --epochs 1 --workers 2 --extra_tag smoke_pvrcnn_veh \
  --skip_builtin_eval --logger_iter_interval 10 --wo_gpu_stat --max_ckpt_save_num 1 \
  --set DATA_CONFIG.DATA_PATH "${BDC_DATA}/claude_canon_all_twin_st_v1" > /tmp/smoke_veh.log 2>&1
echo "rc $? (124 = stopped by the time limit, expected)"
grep -E 'Train:|Traceback|Error' /tmp/smoke_veh.log | tail -n 3 | sed -E 's/.*Train: +/Train: /' | cut -c1-160
