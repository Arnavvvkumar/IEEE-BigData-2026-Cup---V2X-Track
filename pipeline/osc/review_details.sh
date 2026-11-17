#!/bin/bash
# READ-ONLY: why shards failed, why the OSC compile failed, and what the v3 validation produced.
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
cd "$base/osc"
echo "== shard failure reasons (last rsync error line per failed shard, unique)"
for f in $(grep -l SHARD_FAILED upload_shard_*.log | head -n 40); do grep -E 'rsync: |rsync error|ssh_exchange|Connection|Too many|kex_exchange|banner' "$f" | tail -n 1 | cut -c1-140; done | sort | uniq -c | sort -rn | head -n 6
echo "== which shards still running / done"
grep -l SHARD_DONE upload_shard_*.log | sed 's/upload_shard_//;s/.log//' | tr '\n' ' '; echo
echo "== ssh connection limits seen on OSC? (MaxStartups / refused)"
grep -h -i -E 'refused|reset|closed by|too many|maxstartups|exchange' upload_shard_*.log | sort | uniq -c | sort -rn | head -n 5
echo "== OSC build error lines"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=30 <osc_user>@ascend.osc.edu 'R=/fs/scratch/<osc_project>/bdc; grep -n -i -E "error|killed|cannot|nvcc fatal|Terminated|No space|signal" $R/logs/build_env.log | tail -n 14; echo "--- ENV_DONE/BUILD markers"; grep -n -E "ENV_DONE|BUILD_START|BUILD_DONE|numpy |torch |spconv_import_ok" $R/logs/build_env.log | tail -n 8; echo "--- disk"; df -h $R | tail -n 1; ls $R/code/OpenPCDet | head -n 5' 2>/dev/null | grep -v -E '^\*|^\s*$|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials' | cut -c1-200
echo "== v3 queue log tail"
tail -n 18 "$base/final/queue_v3_infer.log" | cut -c1-170
echo "== markers"
cat "$base/final/dev150_v3_ok" "$base/final/dev350_v3_ok" 2>/dev/null; echo
echo "== v3 package + zips"
ls -la "$base/final/candidate_v3" 2>/dev/null | head -n 5; ls -la /mnt/c/Users/arnav/Downloads/*.zip 2>/dev/null | cut -c30-140
