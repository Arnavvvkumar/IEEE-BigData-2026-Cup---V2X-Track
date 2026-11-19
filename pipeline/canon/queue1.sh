#!/bin/bash
# GPU queue 1: ground-canonical SECOND on the relabelled 6k mix.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
bash "$here/run_arm.sh" canon_relabel_s1234 "$here/second_canon_v1.yaml" "$D/claude_canon_relabel_mix6000_v1" 12 '-3.2 -3.0 -2.8'
