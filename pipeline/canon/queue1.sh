#!/bin/bash
# GPU queue 1: ground-canonical SECOND on the relabelled 6k mix.
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
D="$HOME/projects/bdc2026-track-b/data"
bash "$here/run_arm.sh" canon_relabel_s1234 "$here/second_canon_v1.yaml" "$D/claude_canon_relabel_mix6000_v1" 12 '-3.2 -3.0 -2.8'
