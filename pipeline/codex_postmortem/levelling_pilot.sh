#!/usr/bin/env bash
# Local screened-development inference. The sole factor is the levelling plane.
set -euo pipefail
this_dir="$(cd "$(dirname "$0")" && pwd)"
experiment="$(dirname "$this_dir")"
program="$HOME/projects/detector-program"
repo="$program/OpenPCDet"
data="$HOME/projects/bdc2026-track-b/data/claude_canon_all_v1"
python="$program/env310/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

size="${1:?use 150 or 350}"
count="${2:?number of screened scans, or 0 for all}"
[[ "$size" == 150 || "$size" == 350 ]] && [[ "$count" =~ ^[0-9]+$ ]]
checkpoint="$(find "$repo/output" -path '*/canon_all_s1234/ckpt/checkpoint_epoch_12.pth' -print -quit)"
[[ -f "$checkpoint" ]]
(
  cd "$experiment"
  PYTHONPATH="$experiment/codex_review" "$python" -B -c 'from replay import screened; screened()'
)
out="$this_dir/level_dev${size}_n${count}"
mkdir "$out"
scans="$experiment/coop/dev$size/scans"
if (( count > 0 )); then
    "$python" -B - "$experiment/coop/dev$size/index.json" "$scans" "$out/scans" "$count" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

index, source, target = map(Path, sys.argv[1:4])
groups = defaultdict(list)
for key, row in sorted(json.loads(index.read_text()).items()):
    groups[row['scenario']].append(key)
keys = sorted(groups)
choice = np.random.default_rng(1234).choice(keys, int(sys.argv[4]), replace=False)
target.mkdir()
for scenario in sorted(choice):
    frames = groups[scenario]
    name = f'{frames[len(frames) // 2]}.bin'
    (target / name).symlink_to(source / name)
PY
    scans="$out/scans"
fi
cd "$repo/tools"
common=(--repo "$repo" --cfg "$experiment/canon/second_canon_all_v1.yaml" --data "$data"
        --checkpoint "$checkpoint" --scans "$scans"
        --ground -3.15 -3.075 -3.0 -2.925 -2.85 -2.775 --batch 1 --workers 1)
PLANE=median "$python" -B "$experiment/final/infer_test.py" "${common[@]}" --out "$out/median.json" > "$out/median.log" 2>&1
"$python" -B "$experiment/codex_review/per_scan_export.py" infer_test.py "${common[@]}" --out "$out/independent.json" > "$out/independent.log" 2>&1
cd "$experiment"
"$python" -B "$this_dir/score_levelling.py" "$size" "$out" > "$out/ap.jsonl"
