#!/bin/bash
# Local side: is push_code still running, and how big is the OpenPCDet tree we intend to send?
set -uo pipefail
echo "=== processes ==="
ps -o pid,etime,cmd -p 4406,4160,4193 2>/dev/null || true
pgrep -af 'push_code|upload_rsync|rsync' || true
echo "=== push_code2.log ==="
wc -l "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc/push_code2.log"
echo "=== local OpenPCDet (same excludes as push_code.sh) ==="
src="$HOME/projects/detector-program/OpenPCDet"
du -sh "$src" 2>/dev/null || echo "missing $src"
# file count excluding the same paths rsync excludes
find "$src" -type f \
  -not -path '*/output/*' \
  -not -path '*/__pycache__/*' \
  -not -path '*/build/*' \
  -not -path '*/.git/*' \
  -not -name '*.so' \
  -not -path '*.egg-info/*' \
  | wc -l
echo "=== key local files ==="
ls -la "$src/setup.py" "$src/tools/train.py" "$src/pcdet/__init__.py" 2>&1 | head
echo "=== upload log tail ==="
tail -n 8 "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc/upload_rsync.log"
