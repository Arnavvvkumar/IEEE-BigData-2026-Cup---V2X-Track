#!/bin/bash
echo "train $(ps -eo cmd | grep -c 'extra_tag vnx010_twin_s2')"
i=0
while [ "$i" -lt 8 ]; do
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
  i=$((i + 1))
  sleep 1
done
