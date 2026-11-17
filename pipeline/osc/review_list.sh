#!/bin/bash
# READ-ONLY: files created/modified recently by Cursor (since 17:30 today), plus process state.
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
cd "$base"
date
echo "== files modified since 17:30 (excluding big data / checkpoints)"
find experiments CURSOR_HANDOFF_2026-11-11.md -type f -newermt '2026-11-11 17:30' ! -path '*/coop/more*' ! -path '*/coop/dev*/scans/*' ! -name '*.pth' ! -path '*/allpred/*' ! -path '*/test_allpred/*' ! -path '*__pycache__*' 2>/dev/null \
  -printf '%TH:%TM  %6s  %p\n' | sort | cut -c1-200 | head -n 80
echo "== big outputs recently written"
find experiments -type f -newermt '2026-11-11 17:30' \( -path '*/allpred/*' -o -path '*/test_allpred/*' \) -printf '%TH:%TM  %8s  %p\n' 2>/dev/null | sort | cut -c1-200 | head -n 40
echo "== processes"
pgrep -fa 'train.py' | grep -o 'extra_tag [a-z0-9_]*' | sort -u | tr '\n' ' '; echo
echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "rsync shards: $(pgrep -fc 'rsync -aL --partial --size-only --files-from')  other infer/eval: $(pgrep -fa 'infer_test|infer_tta|eval150|build_cand|run_tta|run_infer' | wc -l)"
ls ~/projects/bdc2026-track-b/upload_stage 2>/dev/null | head -n 5
echo "upload_stage files: $(ls ~/projects/bdc2026-track-b/upload_stage 2>/dev/null | wc -l)"
ls -la ~/projects/bdc2026-track-b/ | grep -i -E 'upload|osc|log' | head
