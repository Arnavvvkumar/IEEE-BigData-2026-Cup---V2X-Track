#!/bin/bash
# Training progress + where the time goes (data loading vs GPU), CPU/memory pressure, dataloader settings.
date +%H:%M
grep -E 'Train:' ~/projects/detector-program/second_twin_ll_s1.log | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-210
grep -E 'Train:' ~/projects/detector-program/second_twin_ll_s1.log | tail -n 40 | sed -E 's/.*(Data time: [0-9.()]+ +Forward time: [0-9.()]+ +Batch time: [0-9.()]+).*/\1/' | tail -n 2
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 2; done | tr '\n' ' '; echo "(GPU % samples)"
echo "cores: $(nproc) | load: $(cut -d' ' -f1-3 /proc/loadavg) | $(free -g | sed -n 2p)"
ps -eo pcpu,rss,cmd --sort=-pcpu | grep -E 'python' | grep -v grep | head -n 8 | awk '{printf "%5s%% %5.1fGB  %s\n",$1,$2/1048576,substr($0,index($0,$3),70)}'
echo "--- run_arm_par_ns.sh passes: --workers ${WORKERS:-4}; yaml batch/workers:"
grep -n -E 'BATCH_SIZE_PER_GPU|NUM_WORKERS|MAX_NUMBER_OF_VOXELS|VOXEL_SIZE' "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon/second_canon_all_v1.yaml" | head
echo "--- data on disk: $(df -h ~/projects/bdc2026-track-b/data | tail -n 1 | awk '{print $1, $2, $3, $5}') | dataset files are .npy float32 on the WSL ext4 disk"
