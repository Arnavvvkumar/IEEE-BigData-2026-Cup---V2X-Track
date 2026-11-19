#!/bin/bash
# Truck rule with LOWER copy scores (a = score factor for truck-like vehicle boxes copied as trucks): on the v9 bundle-B ensemble, both dev sets.
# Motivation (label-free, test frames): vehicle-copy truck boxes with scores 0.15-0.30 (0.7 per frame) outrank the large fixed vehicles at spots B and C.
cd "${BDC_ROOT}/experiments/claude_carla_overnight/coop"
P="${BDC_PYTHON}"
for s in dev150 dev350; do
  echo "=== $s $(date -Is)"
  "$P" rule_retune.py $s v9B_norule 4 'a 0.5 base 0.1,lowA'
done
echo LOWA_DONE
