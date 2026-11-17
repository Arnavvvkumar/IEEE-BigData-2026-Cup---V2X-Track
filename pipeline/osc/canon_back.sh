#!/bin/bash
echo "canon_rsync $(ps -eo cmd | grep 'rsync -aL' | grep -c 'claude_canon_all_v1.shard')"
ps -eo etime,cmd | grep 'rsync -aL' | grep 'claude_canon_all_v1.shard' | grep -v grep | sed 's/ -e ssh.*//' | head -n 10
echo "train $(ps -eo cmd | grep train.py | grep -c second_twin_st_s1)"
