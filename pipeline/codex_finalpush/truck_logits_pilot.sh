#!/usr/bin/env bash
# Matched small diagnostic on screened dev, using an existing SECOND voter.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
exp="$(dirname "$here")"
program="$HOME/projects/detector-program"
repo="$program/OpenPCDet"
python="$program/env310/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$exp/codex_review" "$python" -B -c 'from replay import screened; screened()'

checkpoint="$(find "$repo/output" -path '*/second_twin_st_s1/ckpt/checkpoint_epoch_12.pth' -print -quit)"
[[ -f "$checkpoint" ]]
limit="${1:-24}"
[[ "$limit" =~ ^[0-9]+$ ]]
for size in 150 350; do
    out="$here/truck_logits_dev$size"
    if (( limit != 24 )); then
        out="${out}_n$limit"
    fi
    mkdir "$out"
    for mode in stock retain; do
        cd "$repo/tools"
        used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
        (( used < 2048 ))
        TRUCK_LOGIT_MODE="$mode" "$python" -B "$here/truck_logits_export.py" \
            --repo "$repo" --cfg "$exp/canon/second_canon_all_v1.yaml" \
            --data "$HOME/projects/bdc2026-track-b/data/claude_canon_all_twin_st_v1" \
            --checkpoint "$checkpoint" --scans "$exp/coop/dev$size/scans" \
            --out "$out/$mode.json" --ground -3.0 --limit "$limit" --workers 1 > "$out/$mode.log" 2>&1
    done
    cd "$exp"
    "$python" -B "$here/truck_logits_score.py" "$size" "$out" > "$out/results.jsonl"
done
