#!/bin/bash
# Read-only: everything still running on the laptop.
date +%H:%M
echo "--- processes"
for p in 'train.py' 'infer_test.py' 'eval150.py' 'rule_retune|hedge2|verify_hedge|rider_test|ped_hedge' 'p12_combine' 'p13_seed2' 'globusconnect' 'scp ' 'rsync'; do
  echo "$p: $(pgrep -fc "$p")"
done
echo "--- combined run (tail)"
tail -n 4 "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final/p12_combine.log" | cut -c1-200
echo "--- trainings"
for t in second_twin_ll_s2 vnx010_pv_ft4; do echo "$t: $(grep -E 'Train:' ~/projects/detector-program/$t.log 2>/dev/null | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-70)"; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; free -g | sed -n 2p
~/gcp/globusconnectpersonal -status 2>/dev/null | tr '\n' ' '; echo
