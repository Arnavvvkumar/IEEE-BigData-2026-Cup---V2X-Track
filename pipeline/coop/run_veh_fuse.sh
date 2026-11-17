#!/bin/bash
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop"
for s in dev150 dev350; do echo "== $s $(date -Is)"; ~/projects/detector-program/env310/bin/python veh_fuse_test.py $s 2>&1 | tail -n 10; done
echo VEH_FUSE_DONE
