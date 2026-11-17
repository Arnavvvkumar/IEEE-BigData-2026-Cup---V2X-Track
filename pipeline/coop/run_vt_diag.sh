#!/bin/bash
# CPU-only: why vehicles and trucks lose AP (by-type outcomes, then oracle decomposition per class) on dev350.
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop"
P="$HOME/projects/detector-program/env310/bin/python"
echo "VT_START $(date -Is)"
"$P" cls_types.py dev350 pred_p4_addV2ped.json
echo "=== vehicle oracle"; "$P" cls_oracle.py dev350 pred_p4_addV2ped.json vehicle
echo "=== truck oracle";   "$P" cls_oracle.py dev350 pred_p4_addV2ped.json truck
echo "VT_DONE $(date -Is)"
