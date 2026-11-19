#!/bin/bash
cd "${BDC_ROOT}/experiments/claude_carla_overnight/coop"
for s in dev350 dev150; do echo "== $s"; ${BDC_PYTHON} dedupe_test.py $s pred_p4_addV2ped.json 2>&1 | tail -n 7; done
