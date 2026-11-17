#!/bin/bash
# GPU: train one SECOND arm, then score it on the real dev frames with test-time ground levelling.
# usage: run_arm.sh <tag> <cfg yaml> <dataset dir> <epochs> <ground z list, quoted> [extra train args...]
set -euo pipefail
tag=$1; config=$2; dataset=$3; epochs=$4; grounds=$5; shift 5
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
program="$HOME/projects/detector-program"
repo="$program/OpenPCDet"
python="$program/env310/bin/python"
ea="$root/experiments/claude_carla_overnight/erroranalysis"
res="$root/experiments/claude_carla_overnight/canon/results"
mkdir -p "$res"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while pgrep -af '[p]ython.*train\.py' | grep -q env310; do sleep 30; done
cd "$repo/tools"
"$python" -u train.py --cfg_file "$config" --batch_size "${BS:-2}" --epochs "$epochs" --workers 8 --extra_tag "$tag" --fix_random_seed \
  --skip_builtin_eval --logger_iter_interval 50 --wo_gpu_stat --max_ckpt_save_num 6 --set DATA_CONFIG.DATA_PATH "$dataset" "$@" > "$program/$tag.log" 2>&1
ckdir=$(dirname "$(find "$repo/output" -path "*/$tag/ckpt/checkpoint_epoch_${epochs}.pth" -print -quit)")
for e in $epochs $((epochs-1)) $((epochs-2)); do
  "$python" "$ea/detilt_eval.py" --repo "$repo" --cfg "$config" --data "$dataset" --checkpoint "$ckdir/checkpoint_epoch_${e}.pth" \
    --out "$res/${tag}_ep${e}.json" --plane median --ground $grounds --save-best 2>&1 | grep -E '^median|Error|Traceback' | sed "s/^/$tag ep$e /" | tee -a "$res/summary.txt"
done
