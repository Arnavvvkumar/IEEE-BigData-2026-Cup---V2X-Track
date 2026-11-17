#!/bin/bash
# v11 = v10r + extra voter passes (ADD) [+ optional truck copy factor TA].  Nothing is submitted.
# usage: ADD="pvrcnn_ped_P2_ep5:pedestrian pvrcnn_veh_V1_ep5:vehicle" [TA=0.3] bash build_v11.sh <name e.g. 11 or 11t2>
set -euo pipefail
name=$1
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final"
g="$here/_build_${name}_gen.sh"
sed -e "s/candidate_v10r/candidate_v$name/g" -e "s/submission_v10r\.zip/submission_v$name.zip/g" -e "s/v10r-realism-2026-11-13/v$name-final-2026-11-14/" \
    -e "s/V10R_BUILT/V${name}_BUILT/" -e "s#_declaration_backup_r.txt#_declaration_backup_$name.txt#g" \
    -e "s#\$P1 \$P1 \$P1\"#\$P1 \$P1 \$P1 ${ADD:-}\"#" "$here/_build_v10r_gen.sh" > "$g"
sed -i 's#^sed -i "s/^Submission tag:#sed -i "s/^Date (YYYY-MM-DD):.*/Date (YYYY-MM-DD):      2026-11-14/" "$dl/declaration.txt"; &#' "$g"
[ -n "${TA:-}" ] && sed -i "s/^export FUSE_ONE_VOTE=vehicle/export FUSE_ONE_VOTE=vehicle; export TRUCK_A=$TA/" "$g"
sed -i "s#^export FUSE_ONE_VOTE=vehicle#export FUSE_ONE_VOTE=vehicle; export TESTPRED=\"\$here/test_allpred_ps1\" DROP=second_twin_ll_s1_ep12#" "$g"
grep -E 'EXTRA=|TRUCK_A|TESTPRED' "$g" | cut -c1-250
sed -i "s/final_synthetic_v4/final_synthetic_v3/g" "$g"; grep -c final_synthetic_v3 "$g"; bash "$g" 2>&1 | grep -E 'spec|rule|zip |scorer|BUILT|Traceback|Error|missing' | cut -c1-250
SYN=final_synthetic_v3 ~/projects/detector-program/env310/bin/python "$here/verify_zip_any.py" "bdc2026_trackB_submission_v$name.zip" bdc2026_trackB_submission_v10r.zip | grep -E 'detections|finite|truck|declaration|synthetic'
