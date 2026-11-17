#!/bin/bash
# Read-only upload progress. Does not interrupt upload_rsync.sh.
set -uo pipefail
echo "=== local ==="
pgrep -af 'upload_rsync|rsync -aL' || echo "uploader not running"
echo "uploader_count $(pgrep -fc upload_rsync.sh || true)"
tail -n 6 "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc/upload_rsync.log"
echo "=== remote du ==="
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=40 <osc_user>@ascend.osc.edu \
  'du -sm /fs/scratch/<osc_project>/bdc/data /fs/scratch/<osc_project>/bdc/data/* 2>/dev/null; echo ---; find /fs/scratch/<osc_project>/bdc/data/claude_twin_v1/points -name "twin_*.npy" 2>/dev/null | wc -l; find /fs/scratch/<osc_project>/bdc/data/claude_canon_all_v1 -maxdepth 1 -type d 2>/dev/null | wc -l' \
  2>/dev/null | grep -v -E '^\*|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials|^$'
