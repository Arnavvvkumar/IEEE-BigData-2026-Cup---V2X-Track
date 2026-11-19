#!/bin/bash
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
bash "$here/eval_arm.sh" canon_relabel_s1234 "$here/second_canon_v1.yaml" "${BDC_DATA}/claude_canon_relabel_mix6000_v1" '12 11 10' '-3.3 -3.15 -3.0 -2.85 -2.7'
