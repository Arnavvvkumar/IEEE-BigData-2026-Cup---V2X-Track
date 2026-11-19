#!/bin/bash
# GPU: one SECOND arm on the relabelled 5k Generator V2 + 1k CARLA (+containers) mix. Same trainer flags, seed, epochs and
# evaluator as Codex's matched overnight runs (run_second_overnight.sh); only the dataset and the label-derived anchors differ.
# Refuses to start if another OpenPCDet job is running. ~75 min train + ~5 min eval on the RTX 4070 laptop GPU.
set -euo pipefail
root="${BDC_ROOT}"
program="${BDC_PROGRAM}"
repo="$program/OpenPCDet"
python="$program/env310/bin/python"
data="${BDC_DATA}"
config="$root/experiments/claude_carla_overnight/boxsize/second_relabel_v1.yaml"
dataset="$data/claude_v2carla_relabel_mix6000_v1"
tag=relabel_containers_seed1234
out="$root/experiments/detector_program/overnight_accuracy_results"
log="$program/second_${tag}.log"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if pgrep -af '[p]ython.*(train\.py|evaluate_openpcdet\.py)' | grep -q env310; then echo "GPU busy"; exit 1; fi
cd "$repo/tools"
"$python" -u train.py --cfg_file "$config" --batch_size 2 --epochs 12 --workers 6 --extra_tag $tag --fix_random_seed \
  --skip_builtin_eval --logger_iter_interval 25 --wo_gpu_stat --set DATA_CONFIG.DATA_PATH "$dataset" > "$log" 2>&1
ckpt=$(find "$repo/output" -path "*/$tag/ckpt/checkpoint_epoch_12.pth" -print -quit)
common=(--repo "$repo" --cfg "$config" --data "$dataset" --checkpoint "$ckpt" --source "$data/claude_v2_scale5500_fix1"
        --evaluator "$root/starting_kit/track_b/scoring_program/detection_eval.py")
"$python" "$root/experiments/detector_program/evaluate_openpcdet.py" "${common[@]}" --out "$out/${tag}_synthetic"
"$python" "$root/experiments/detector_program/evaluate_openpcdet.py" "${common[@]}" --out "$out/${tag}_real" \
  --real-kit "$root/experiments/real_reference_audit/real_dev_cw1"
cd "$root"
${BDC_PYTHON} experiments/claude_carla_overnight/boxsize/score_halves.py \
  "$out/overnight_mix_seed1234_real/predictions.json" "$out/${tag}_real/predictions.json" | tee "$out/${tag}_halves.txt"
