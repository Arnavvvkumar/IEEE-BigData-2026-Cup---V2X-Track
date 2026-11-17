#!/bin/bash
echo "=== rsync ==="
ps -eo etime,pid,cmd | grep 'rsync -aL' | grep -v grep | sed 's/ -e ssh.*//'
echo "=== shard parents ==="
ps -eo etime,pid,cmd | grep 'upload_shard.sh claude' | grep -v grep
echo "=== local points ==="
echo "twin $(ls "$HOME/projects/bdc2026-track-b/data/claude_twin_v1/points" | wc -l)"
echo "canon $(ls "$HOME/projects/bdc2026-track-b/data/claude_canon_all_v1/points" | wc -l)"
echo "=== local GB ==="
du -sm "$HOME/projects/bdc2026-track-b/data/claude_twin_v1" "$HOME/projects/bdc2026-track-b/data/claude_canon_all_v1"
