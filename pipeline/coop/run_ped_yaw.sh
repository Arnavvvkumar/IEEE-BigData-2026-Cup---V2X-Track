#!/bin/bash
cd "${BDC_ROOT}/experiments/claude_carla_overnight/coop"
P="${BDC_PYTHON}"
for s in dev150 dev350; do "$P" ped_yaw_hedge.py $s pf_yawcirc_norule 3; done
