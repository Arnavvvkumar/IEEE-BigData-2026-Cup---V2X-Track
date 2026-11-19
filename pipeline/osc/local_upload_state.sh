#!/bin/bash
echo "=== rsync ==="
ps -eo etime,pid,cmd | grep 'rsync -aL' | grep -v grep | sed 's/ -e ssh.*//'
echo "=== shard parents ==="
ps -eo etime,pid,cmd | grep 'upload_shard.sh claude' | grep -v grep
echo "=== local points ==="
echo "twin $(ls "${BDC_DATA}/claude_twin_v1/points" | wc -l)"
echo "canon $(ls "${BDC_DATA}/claude_canon_all_v1/points" | wc -l)"
echo "=== local GB ==="
du -sm "${BDC_DATA}/claude_twin_v1" "${BDC_DATA}/claude_canon_all_v1"
