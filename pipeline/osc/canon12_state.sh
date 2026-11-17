#!/bin/bash
echo "canon_rsync $(ps -eo cmd | grep -c 'claude_canon_all_v1.shard' || true)"
echo "shard_bash $(ps -eo cmd | grep -c 'upload_shard.sh claude_canon' || true)"
echo "six $(ps -eo cmd | grep -c 'upload_six.sh' || true)"
echo "train $(ps -eo cmd | grep -c 'extra_tag vnx010_twin_s2' || true)"
ps -eo etime,cmd | grep 'rsync -aL' | grep -v grep | sed 's/ -e ssh.*//' | head -n 15
