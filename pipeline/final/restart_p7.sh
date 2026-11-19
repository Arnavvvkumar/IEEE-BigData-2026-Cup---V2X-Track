#!/bin/bash
# Stop the serial P7 run (keeps any finished prediction file) so the parallel version can take over.
pkill -f p7_cheap.sh
pkill -f rule_retune.py
pkill -f rider_test.py
pkill -f 'eval150.py dev... v5_norule'
sleep 3
cd "${BDC_ROOT}/experiments/claude_carla_overnight"
cat final/p7_cheap.log | cut -c1-200
ls -la coop/dev150/pred_v5_norule.json coop/dev350/pred_v5_norule.json 2>&1 | cut -c1-160
free -g | sed -n 2p
