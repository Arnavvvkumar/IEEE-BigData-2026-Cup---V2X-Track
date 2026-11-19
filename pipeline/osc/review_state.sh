#!/bin/bash
# READ-ONLY review of the current state of Cursor's work. Changes nothing.
base="${BDC_ROOT}/experiments/claude_carla_overnight"
date
echo "== local shard logs: done / failed / total"
cd "$base/osc"
echo "SHARD_DONE: $(grep -l SHARD_DONE upload_shard_*.log 2>/dev/null | wc -l)  SHARD_FAILED: $(grep -l SHARD_FAILED upload_shard_*.log 2>/dev/null | wc -l)  total logs: $(ls upload_shard_*.log 2>/dev/null | wc -l)"
echo "running shard rsyncs: $(pgrep -fc 'rsync -aL --partial --size-only --files-from')"
echo "== recent final/ files (top level only, last 3 h)"
find "$base/final" -maxdepth 1 -type f -newermt '2026-11-11 17:30' -printf '%TH:%TM  %6s  %f\n' | sort
echo "== final/test_allpred + coop allpred newer than 17:30"
find "$base/final/test_allpred" "$base/coop/dev150/allpred" "$base/coop/dev350/allpred" -maxdepth 1 -type f -newermt '2026-11-11 17:30' -printf '%TH:%TM  %9s  %p\n' 2>/dev/null | sed "s|$base/||" | sort
echo "== trainings"
pgrep -fa train.py | grep -o 'extra_tag [a-z0-9_]*' | sort -u | tr '\n' ' '; echo
tail -c 300 ${BDC_PROGRAM}/vnx010_twin_s2.log | tr '\r' '\n' | tail -n 1 | cut -c1-150
echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "python inference/eval procs: $(pgrep -fa 'infer_test|infer_tta|eval150|run_tta|run_infer' | wc -l)"
echo "== OSC (read-only)"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=30 <osc_user>@ascend.osc.edu 'R=/fs/scratch/<osc_project>/bdc; du -sm $R/data/* 2>/dev/null; echo "--- build log tail"; tail -n 12 $R/logs/build_env.log 2>/dev/null; echo "--- logs dir"; ls -t $R/logs | head -n 8; echo "--- squeue"; squeue -u <osc_user> 2>&1 | head -n 5; echo "--- smoke out tail"; for f in $(ls -t $R/logs/smoke_*.out 2>/dev/null | head -n 1); do tail -n 12 $f; done; echo "--- cfg"; ls $R/cfg | tr "\n" " "' 2>/dev/null | grep -v -E '^\*|^\s*$|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials' | cut -c1-170
