#!/bin/bash
# Stop the pedestrian size-rule grid (it does not help); leave everything else running.
pkill -f run_size_rule.sh
pkill -f ped_size_rule.py
sleep 2
pgrep -af 'ped_size_rule|eval150|train.py' | cut -c1-120
