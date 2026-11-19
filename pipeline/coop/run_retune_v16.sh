#!/bin/bash
# Pedestrian hedge re-tune on the v16 base (averaged pedestrian heading); same pre-registered grid as ped_retune.py.
cd "${BDC_ROOT}/experiments/claude_carla_overnight/coop"
P="${BDC_PYTHON}"
for s in dev150 dev350; do "$P" ped_retune.py $s pf_yawcirc_norule 2; done
