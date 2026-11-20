#!/usr/bin/env bash
# Continue the two matched runs while this WSL session and the PC stay available.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
program="${BDC_PROGRAM}"
repo="$program/OpenPCDet"
python="$program/env310/bin/python"
data="${BDC_DATA}"
config="$root/experiments/detector_program/second_control_v1.yaml"
handoff="$root/experiments/claude_carla_overnight/SECOND_MIX_READY.json"
output="$root/experiments/detector_program/overnight_accuracy_results"
logs="$program/second_overnight_control.log"
deadline=$((SECONDS + 36000))
mkdir -p "$output"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

find_checkpoint() {
  find "$repo/output" -path "*/$1/ckpt/checkpoint_epoch_12.pth" -print -quit
}

wait_for_control() {
  until [[ -n "$(find_checkpoint overnight_control_seed1234)" ]] && grep -q 'Skipping built-in evaluation' "$logs"; do
    if grep -q 'Traceback (most recent call last)' "$logs"; then
      tail -25 "$logs"
      return 1
    fi
    if (( SECONDS >= deadline )); then return 1; fi
    sleep 30
  done
}

evaluate() {
  local tag="$1" dataset="$2" checkpoint
  checkpoint="$(find_checkpoint "$tag")"
  [[ -n "$checkpoint" ]]
  local common=(--repo "$repo" --cfg "$config" --data "$dataset" --checkpoint "$checkpoint"
                --source "$data/claude_v2_scale5500_fix1"
                --evaluator "$root/starting_kit/track_b/scoring_program/detection_eval.py")
  "$python" "$root/experiments/detector_program/evaluate_openpcdet.py" "${common[@]}" --out "$output/${tag}_synthetic"
  "$python" "$root/experiments/detector_program/evaluate_openpcdet.py" "${common[@]}" --out "$output/${tag}_real" \
    --real-kit "$root/experiments/real_reference_audit/real_dev_cw1"
}

echo "Waiting for the control's epoch-12 checkpoint and clean trainer exit."
wait_for_control
echo "Evaluating the control."
evaluate overnight_control_seed1234 "$data/codex_second_control6000"
echo "Waiting for Claude's audited mixed dataset."
until [[ -s "$handoff" ]]; do
  if (( deadline - SECONDS < 7200 )); then
    echo "Mixed arm deferred. Fewer than two hours remain and the audited dataset is unavailable."
    exit 0
  fi
  sleep 60
done
dataset="$("$python" "$root/experiments/detector_program/verify_second_mix.py" \
  "$handoff" "$data/codex_openpcdet_custom_v1")"
if (( deadline - SECONDS < 7200 )); then
  echo "Mixed arm deferred. Fewer than two hours remain."
  exit 0
fi
if pgrep -af '[p]ython.*(train\.py|evaluate_openpcdet\.py)' | grep -q '/env310/bin/python'; then
  echo "Another OpenPCDet job holds the GPU. Mixed arm deferred."
  exit 1
fi
cd "$repo/tools"
echo "Training mixed SECOND from scratch on $dataset"
"$python" -u train.py --cfg_file "$config" --batch_size 2 --epochs 12 --workers 6 \
  --extra_tag overnight_mix_seed1234 --fix_random_seed --skip_builtin_eval --logger_iter_interval 25 --wo_gpu_stat \
  --set DATA_CONFIG.DATA_PATH "$dataset"
evaluate overnight_mix_seed1234 "$dataset"
echo "Matched runs and both evaluations complete."
