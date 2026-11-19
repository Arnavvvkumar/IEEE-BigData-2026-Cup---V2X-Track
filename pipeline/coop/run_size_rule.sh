#!/bin/bash
# Point-extent pedestrian size rule: grid on dev350 then dev150 (CPU only).
cd "${BDC_ROOT}/experiments/claude_carla_overnight/coop"
P="${BDC_PYTHON}"
for s in dev350 dev150; do
  echo "=== $s $(date -Is)"
  "$P" ped_size_rule.py $s st_truckonly
done
