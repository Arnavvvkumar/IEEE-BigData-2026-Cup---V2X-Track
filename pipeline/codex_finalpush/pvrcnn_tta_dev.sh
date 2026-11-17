#!/usr/bin/env bash
# Run screened dev-only exports after the P1 TTA gate passes.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
exp="$(dirname "$here")"
program="$HOME/projects/detector-program"
repo="$program/OpenPCDet"
python="$program/env310/bin/python"
data="$HOME/projects/bdc2026-track-b/data/claude_canon_all_twin_st_v1"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

size="${1:?use 150 or 350}"
shift
(( $# > 0 ))
[[ "$size" == 150 || "$size" == 350 ]]
grep -q '^P29_DONE' "$exp/final/p29_P1tta.log"
"$python" -B "$here/check_p1.py" "$exp/final/p29_P1tta.log"
PYTHONPATH="$exp/codex_review" "$python" -B -c 'from replay import screened; screened()'

guard_gpu() {
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
    if (( used >= 2048 )) || competing_job; then
        echo 'GPU or existing inference queue is active' >&2
        return 1
    fi
}

competing_job() {
    pgrep -f '^(bash|/[^ ]*/bash) ([^ ]*/)?(p29_P1tta|p28_P456|run_infer_one|reexport_perscan_b1)[.]sh( |$)' >/dev/null
}

for tag in "$@"; do
    case "$tag" in
        pvrcnn_ped_P1|pvrcnn_ped_P2|pvrcnn_ped_P3) cfg=pvrcnn_ped.yaml; epoch=5 ;;
        pvrcnn_st_osc) cfg=pvrcnn_all.yaml; epoch=6 ;;
        pvrcnn_st_osc_s2) cfg=pvrcnn_all.yaml; epoch=4 ;;
        *) echo "Unknown v13 pedestrian PV-RCNN voter: $tag" >&2; exit 2 ;;
    esac
    guard_gpu
    ck="$(find "$repo/output" -path "*/$tag/ckpt/checkpoint_epoch_$epoch.pth" -print -quit)"
    [[ -f "$ck" ]]
    out="$here/tta/dev$size"
    mkdir -p "$out"
    target="$out/${tag}_tta_ep${epoch}_allpred.json"
    [[ ! -e "$target" ]]
    (
        cd "$repo/tools"
        "$python" -B "$exp/codex_review/per_scan_export.py" infer_tta.py --repo "$repo" \
            --cfg "$exp/canon/$cfg" --data "$data" --checkpoint "$ck" \
            --scans "$exp/coop/dev$size/scans" --out "$target" --workers 1 \
            --ground -3.075 -3.0 -2.925 -2.85 \
            --variants 0.05,0.05,0 0.05,0,7 0,0.05,-7 0.025,0.075,14 0.075,0.025,-14 &
        job=$!
        trap 'kill "$job" 2>/dev/null || true' EXIT
        while kill -0 "$job" 2>/dev/null; do
            if competing_job; then
                echo 'Existing inference queue started. Stopping this export.' >&2
                exit 1
            fi
            sleep 5
        done
        wait "$job"
        trap - EXIT
    ) > "$target.log" 2>&1
done
