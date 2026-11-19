#!/bin/bash
# GPU: score every saved epoch checkpoint on the 50 public real dev frames (evaluation only, no training).
# Skips checkpoints already scored. Order: most promising first.
set -uo pipefail
root="${BDC_ROOT}"
program="${BDC_PROGRAM}"
repo="$program/OpenPCDet"
python="$program/env310/bin/python"
data="${BDC_DATA}"
out="$root/experiments/claude_carla_overnight/modelsel/real"
ev="$root/experiments/detector_program/evaluate_openpcdet.py"
mkdir -p "$out"
export OMP_NUM_THREADS=2
dp="$root/experiments/detector_program"
bx="$root/experiments/claude_carla_overnight/boxsize"
o="$repo/output"
# tag | cfg | dataset | ckpt dir | epochs
runs=(
"mix|$dp/second_control_v1.yaml|$data/claude_v2carla_mix6000_v1|$o${BDC_ROOT}/experiments/detector_program/second_control_v1/overnight_mix_seed1234/ckpt|12 11 10 9 8 7 6 5 4 3"
"vnx_b8|$repo/tools/cfgs/custom_models/voxelnext_v1.yaml|$data/codex_openpcdet_custom_v1|$o/custom_models/voxelnext_v1/codex_batch8_from_epoch6/ckpt|20 18 16 14 12 10 8"
"vnx_xy010|$repo/tools/cfgs/custom_models/voxelnext_xy010_v1.yaml|$data/codex_openpcdet_custom_v1|$o/custom_models/voxelnext_xy010_v1/codex_xy010_full_seed1234/ckpt|20 18 16 14 12 10 8 6"
"relabel|$bx/second_relabel_v1.yaml|$data/claude_v2carla_relabel_mix6000_v1|$o${BDC_ROOT}/experiments/claude_carla_overnight/boxsize/second_relabel_v1/relabel_containers_seed1234/ckpt|12 11 10 9 8 6 4"
"control|$dp/second_control_v1.yaml|$data/codex_openpcdet_custom_v1|$o${BDC_ROOT}/experiments/detector_program/second_control_v1/overnight_control_seed1234/ckpt|12 10 8 6 4"
"vnx_full|$repo/tools/cfgs/custom_models/voxelnext_v1.yaml|$data/codex_openpcdet_custom_v1|$o/custom_models/voxelnext_v1/codex_full_seed1234/ckpt|6 4"
)
cd "$repo/tools"
for r in "${runs[@]}"; do
  IFS='|' read -r tag cfg ds ck eps <<< "$r"
  for e in $eps; do
    d="$out/${tag}_ep${e}"
    [ -f "$d/result.json" ] && continue
    rm -rf "$d"
    echo "== $tag ep$e $(date +%T)"
    "$python" "$ev" --repo "$repo" --cfg "$cfg" --data "$ds" --checkpoint "$ck/checkpoint_epoch_${e}.pth" \
      --source "$data/claude_v2_scale5500_fix1" --evaluator "$root/starting_kit/track_b/scoring_program/detection_eval.py" \
      --out "$d" --real-kit "$root/experiments/real_reference_audit/real_dev_cw1" > "$out/${tag}_ep${e}.log" 2>&1 || echo "FAILED $tag ep$e"
  done
done
echo ALL_DONE
