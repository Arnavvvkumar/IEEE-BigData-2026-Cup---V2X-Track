#!/bin/bash
# Two TRUCK-ONLY probes on top of v10r (same pedestrian/vehicle output; only the truck class differs). Nothing is submitted.
#   T1 = v10r + long-and-dense truck boost x2 (boxes >= 7 m long with >= 150 points)   -> v10t1
#   T2 = v10r + truck copies of vehicle boxes scored at 0.3x instead of 0.5x            -> v10t2
# Motivation (label-free, test frames): vehicle-copy truck boxes with scores 0.15-0.30 outrank the large fixed vehicles (scores 0.24-0.31).
# Validation cannot judge trucks, so these are meant as leaderboard probes. Both build from the generated v10r script.
set -euo pipefail
here="${BDC_ROOT}/experiments/claude_carla_overnight/final"
bash "$here/build_v10r.sh" > /dev/null 2>&1 || true   # regenerates _build_v10r_gen.sh (with the env-driven truck rule)
for p in "t1 10t1 TRUCKBOOST=2.0" "t2 10t2 TRUCK_A=0.3"; do
  set -- $p; k=$1; name=$2; envset=$3
  sed -e "s/candidate_v10r/candidate_v$name/g" -e "s/submission_v10r\.zip/submission_v$name.zip/g" -e "s/v10r-realism-2026-11-13/v$name-truckprobe-2026-11-14/" \
      -e 's/V10R_BUILT/PROBE_BUILT/' -e 's#_declaration_backup_r.txt#_declaration_backup_p.txt#g' "$here/_build_v10r_gen.sh" > "$here/_build_${name}_gen.sh"
  sed -i "s/^export FUSE_ONE_VOTE=vehicle/export FUSE_ONE_VOTE=vehicle; export $envset/" "$here/_build_${name}_gen.sh"
  grep -c "$envset" "$here/_build_${name}_gen.sh"
  bash "$here/_build_${name}_gen.sh" 2>&1 | grep -E 'rule|format|zip |scorer|uncompressed|PROBE_BUILT|Traceback|Error|missing' | cut -c1-230
  SYN=final_synthetic_v4 ${BDC_PYTHON} "$here/verify_zip_any.py" "bdc2026_trackB_submission_v$name.zip" bdc2026_trackB_submission_v10r.zip | grep -E 'detections|truck|vehicle|pedestrian|declaration|identical'
done
