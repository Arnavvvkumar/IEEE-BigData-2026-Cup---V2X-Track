#!/bin/bash
# Read-only: who is using the laptop GPU. Does not touch OSC or upload_stage.
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== train/infer ==="
pgrep -af 'train.py|infer_tta|infer_test|detilt_eval' || echo none
echo "=== tta outputs ==="
ls -lh "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final/test_allpred/"*_tta_* 2>/dev/null || echo "no test tta yet"
ls "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop/dev350/allpred" 2>/dev/null | head || echo "no dev350 allpred"
