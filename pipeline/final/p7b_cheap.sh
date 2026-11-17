#!/bin/bash
# P7b: same tests as p7_cheap.sh but the grids run in parallel on CPU cores (the AP scorer is CPU-only).
# Reuses pred_v5_norule.json if the earlier run already wrote it.
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle
V5="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian vnx010_twin_s2_ep12:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck vnx010_pv_ft_ep5:vehicle second_twin_pv_s1_ep14:truck vnx010_pv_ft2_ep5:pedestrian,vehicle"
for s in dev150 dev350; do
  if [ ! -f "$s/pred_v5_norule.json" ]; then
    echo "=== $s norule $(date -Is)"
    # shellcheck disable=SC2086
    "$P" eval150.py $s v5_norule $V5 norule
  fi
  echo "=== $s rule retune $(date -Is)"
  "$P" rule_retune.py $s v5_norule 5
  echo "=== $s riders $(date -Is)"
  "$P" rider_test.py $s ft2_pedveh 6
done
echo "P7_DONE $(date -Is)"
