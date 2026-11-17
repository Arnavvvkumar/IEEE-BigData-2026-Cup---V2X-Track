#!/bin/bash
# BUSY if any OpenPCDet / project GPU python job is present in WSL (Codex reclaiming the GPU), else IDLE.
# (Does not use nvidia-smi process lists, which may include the Windows CARLA render process.)
if pgrep -af '[p]ython.*(train\.py|evaluate_openpcdet\.py|test\.py|claude_utv2x_research_runner|utmix_gpu_pair)' >/dev/null; then
  echo "BUSY $(pgrep -af '[p]ython.*(train\.py|evaluate_openpcdet\.py|test\.py)' | head -1 | cut -c1-120)"; exit
fi
echo IDLE
