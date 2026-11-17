#!/bin/bash
# v10r = v10 detections (unchanged) + the new synthetic scans twin/final_synthetic_v4 (range-matched thinning + in-box zero share 30.8%).
# Generated from build_v10.sh; writes candidate_v10r and bdc2026_trackB_submission_v10r.zip only (v10 and its zip are NOT touched).
set -euo pipefail
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final"
sed -e 's/candidate_v10/candidate_v10r/g' -e 's/submission_v10\.zip/submission_v10r.zip/g' -e 's/v10-p1-onevote-2026-11-13/v10r-realism-2026-11-13/' \
    -e 's#twin/final_synthetic_v3/synthetic#twin/final_synthetic_v4/synthetic#' -e 's/V10_BUILT/V10R_BUILT/' \
    -e 's#_declaration_backup.txt#_declaration_backup_r.txt#g' "$here/build_v10.sh" > "$here/_build_v10r_gen.sh"
grep -c 'final_synthetic_v4' "$here/_build_v10r_gen.sh"
bash "$here/_build_v10r_gen.sh" 2>&1 | grep -E 'rule|format|zip |scorer|uncompressed|CANDIDATE|V10R_BUILT|synthetic from|Traceback|Error|missing'
