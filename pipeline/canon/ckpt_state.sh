#!/bin/bash
# Read-only: latest checkpoints and last log lines of the given training tags.
out="$HOME/projects/detector-program/OpenPCDet/output"
for t in "$@"; do
  f=$(find "$out" -path "*/$t/ckpt/*.pth" -print -quit)
  d=$(dirname "$f")
  echo "== $t  $d"
  ls -la --time-style=+%m-%d_%H:%M "$d" | tail -n 5 | awk '{print $6, $7}'
done
grep -c 'vnx010_twin_s1\|vnx010_all_rs3' "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon/results/summary.txt"
cat "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon/run_arm_par_ns.sh"
