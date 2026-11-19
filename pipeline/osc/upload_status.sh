#!/bin/bash
cd "${BDC_ROOT}/experiments/claude_carla_overnight/osc"
cat upload_six.log
echo "rsync procs: $(pgrep -fc 'rsync -aL')"
echo "shards done: $(cat upload_shard_*.log 2>/dev/null | grep -c SHARD_DONE) failed: $(cat upload_shard_*.log 2>/dev/null | grep -c SHARD_FAILED)"
timeout 60 ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes <osc_user>@ascend.osc.edu 'ls /fs/scratch/<osc_project>/bdc/data/claude_twin_v1/points | wc -l; ls /fs/scratch/<osc_project>/bdc/data/claude_canon_all_v1/points | wc -l' 2>/dev/null | grep -v '^\*'
echo "local: twin $(ls ${BDC_DATA}/claude_twin_v1/points | wc -l) canon $(ls ${BDC_DATA}/claude_canon_all_v1/points | wc -l)"
