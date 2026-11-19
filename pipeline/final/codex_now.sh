#!/bin/bash
# Read-only: what is Codex running / writing right now?
date +%H:%M
echo "--- Codex processes"; pgrep -af 'codex_final' | cut -c1-170
d="${BDC_ROOT}/experiments/codex_final"
echo "--- newest files"; ls -t "$d" | head -n 6 | while read f; do echo "$(stat -c '%y' "$d/$f" | cut -c12-19)  $f"; done
for f in ped_tta.log; do echo "--- $f (tail)"; grep -v -E 'Warning|warn|meshgrid|IntTensor|conv2d|^\s*$' "$d/$f" | tail -n 6 | cut -c1-200; done
echo "--- GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- my OneVote test"; tail -n 3 "${BDC_ROOT}/experiments/claude_carla_overnight/coop/onevote.log" | cut -c1-210
