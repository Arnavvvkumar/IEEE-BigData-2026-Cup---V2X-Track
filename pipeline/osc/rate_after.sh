#!/bin/bash
echo "train $(ps -eo cmd | grep 'train.py' | grep -c 'vnx010_twin_s2')"
echo "upload_six $(ps -eo cmd | grep 'upload_six.sh' | grep -v grep | wc -l)"
echo "canon_rsync $(ps -eo cmd | grep 'rsync -aL' | grep -c 'claude_canon_all_v1.shard')"
bash "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc/parallel_rate.sh"
