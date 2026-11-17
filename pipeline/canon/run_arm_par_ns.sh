#!/bin/bash
# Same as run_arm_par.sh but WITHOUT --fix_random_seed, so each run gets a different random seed (ensemble members / seed-variance check).
# usage: [BS=2 WORKERS=4] run_arm_par_ns.sh <tag> <cfg yaml> <dataset dir> <epochs> <ground z list, quoted> [extra train args...]
set -euo pipefail
tag=$1; config=$2; dataset=$3; epochs=$4; grounds=$5; shift 5
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
program="$HOME/projects/detector-program"
repo="$program/OpenPCDet"
python="$program/env310/bin/python"
here="$root/experiments/claude_carla_overnight/canon"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$repo/tools"
"$python" -u train.py --cfg_file "$config" --batch_size "${BS:-2}" --epochs "$epochs" --workers "${WORKERS:-4}" --extra_tag "$tag" \
  --skip_builtin_eval --logger_iter_interval 50 --wo_gpu_stat --max_ckpt_save_num 4 "$@" --set DATA_CONFIG.DATA_PATH "$dataset" > "$program/$tag.log" 2>&1
bash "$here/eval_arm.sh" "$tag" "$config" "$dataset" "$epochs $((epochs-1)) $((epochs-2))" "$grounds"
