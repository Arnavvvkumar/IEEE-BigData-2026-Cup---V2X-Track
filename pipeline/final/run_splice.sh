#!/bin/bash
# usage: run_splice.sh <ped-source version> <out version>   (vehicle+truck always from v13)
F="${BDC_ROOT}/experiments/claude_carla_overnight/final"
cd ${BDC_OUT}
python3 "$F/splice_ped.py" bdc2026_trackB_submission_v13.zip bdc2026_trackB_submission_v$1.zip bdc2026_trackB_submission_v$2.zip v$2-final
bash "$F/kit_validate.sh" bdc2026_trackB_submission_v$2.zip 2>&1 | grep -E 'valid_submission|score_realism'
