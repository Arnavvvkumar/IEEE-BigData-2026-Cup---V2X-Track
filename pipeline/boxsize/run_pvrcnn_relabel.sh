#!/bin/bash
# GPU: PV-RCNN scratch batch-1 on the relabelled 5k Generator V2 + 1k CARLA (+containers) mix, then the official evaluator on
# synthetic val and the 50 real dev frames (+ halves vs the SECOND mix). Usage: run_pvrcnn_relabel.sh smoke|full
set -euo pipefail
mode=${1:?smoke|full}
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
program="$HOME/projects/detector-program"
repo="$program/OpenPCDet"
python="$program/env310/bin/python"
data="$HOME/projects/bdc2026-track-b/data"
config="$root/experiments/claude_carla_overnight/boxsize/pvrcnn_relabel_b1_v1.yaml"
dataset="$data/claude_v2carla_relabel_mix6000_v1"
out="$root/experiments/detector_program/overnight_accuracy_results"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if pgrep -af '[p]ython.*(train\.py|evaluate_openpcdet\.py)' | grep -q env310; then echo "GPU busy"; exit 1; fi
cd "$repo/tools"
if [ "$mode" = smoke ]; then
  tag=pvrcnn_relabel_smoke
  timeout 600 "$python" -u train.py --cfg_file "$config" --batch_size 1 --epochs 1 --workers 6 --extra_tag $tag --fix_random_seed \
    --skip_builtin_eval --logger_iter_interval 25 --wo_gpu_stat --set DATA_CONFIG.DATA_PATH "$dataset" > "$program/$tag.log" 2>&1 || true
  exit 0
fi
tag=pvrcnn_relabel_seed1234
"$python" -u train.py --cfg_file "$config" --batch_size 1 --epochs 12 --workers 6 --extra_tag $tag --fix_random_seed \
  --skip_builtin_eval --logger_iter_interval 50 --wo_gpu_stat --set DATA_CONFIG.DATA_PATH "$dataset" > "$program/$tag.log" 2>&1
ckpt=$(find "$repo/output" -path "*/$tag/ckpt/checkpoint_epoch_12.pth" -print -quit)
common=(--repo "$repo" --cfg "$config" --data "$dataset" --checkpoint "$ckpt" --source "$data/claude_v2_scale5500_fix1"
        --evaluator "$root/starting_kit/track_b/scoring_program/detection_eval.py")
"$python" "$root/experiments/detector_program/evaluate_openpcdet.py" "${common[@]}" --out "$out/${tag}_synthetic"
"$python" "$root/experiments/detector_program/evaluate_openpcdet.py" "${common[@]}" --out "$out/${tag}_real" \
  --real-kit "$root/experiments/real_reference_audit/real_dev_cw1"
cd "$root"
~/projects/bdc2026-track-b/.env/bin/python experiments/claude_carla_overnight/boxsize/score_halves.py \
  "$out/overnight_mix_seed1234_real/predictions.json" "$out/relabel_containers_seed1234_real/predictions.json" \
  "$out/${tag}_real/predictions.json" | tee "$out/${tag}_halves.txt"
