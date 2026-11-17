#!/bin/bash
# Stop the truck height test (answer is already clear: no gain on genuine trucks) so the queued P7 tests start.
pkill -f truck_h_test.py
sleep 25
tail -n 3 "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final/p7_cheap.log"
pgrep -af 'eval150|rule_retune|rider_test' | cut -c1-100
