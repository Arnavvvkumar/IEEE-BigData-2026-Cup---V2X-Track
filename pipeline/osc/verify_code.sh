#!/bin/bash
# Read-only completeness check of the code that landed on OSC. Does not touch the data upload.
set -uo pipefail
src="$HOME/projects/detector-program/OpenPCDet"
echo "=== local file count (rsync excludes) ==="
find "$src" -type f \
  -not -path '*/output/*' \
  -not -path '*/__pycache__/*' \
  -not -path '*/build/*' \
  -not -path '*/.git/*' \
  -not -name '*.so' \
  -not -path '*.egg-info/*' | wc -l
echo "=== local key files ==="
for f in setup.py tools/train.py pcdet/__init__.py pcdet/ops/iou3d_nms/src/iou3d_nms.cpp requirements.txt; do
  if [ -f "$src/$f" ]; then echo "local ok $f"; else echo "local MISSING $f"; fi
done
echo "=== remote ==="
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=40 <osc_user>@ascend.osc.edu 'bash -s' << 'REMOTE' 2>/dev/null | grep -v -E '^\*|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials|^$'
set -uo pipefail
R=/fs/scratch/<osc_project>/bdc
echo "remote file count:"
find "$R/code/OpenPCDet" -type f | wc -l
echo "remote bytes:"
du -sb "$R/code/OpenPCDet" "$R/cfg"
echo "key files:"
for f in setup.py tools/train.py pcdet/__init__.py pcdet/ops/iou3d_nms/src/iou3d_nms.cpp requirements.txt; do
  if [ -f "$R/code/OpenPCDet/$f" ]; then echo "remote ok $f"; else echo "remote MISSING $f"; fi
done
echo "cfg required:"
for f in merge_sets.py run_arm_par_ns.sh requirements_laptop.txt vnx010_all.yaml second_canon_all_v1.yaml; do
  if [ -f "$R/cfg/$f" ]; then echo "cfg ok $f"; else echo "cfg MISSING $f"; fi
done
echo "yaml count: $(ls -1 "$R/cfg"/*.yaml 2>/dev/null | wc -l)"
echo "data:"
du -sm "$R/data"/* 2>/dev/null || echo "no data dirs"
echo "data file sample:"
find "$R/data" -maxdepth 2 -type d | head -n 20
REMOTE
