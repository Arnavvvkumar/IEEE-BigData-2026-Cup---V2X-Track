#!/bin/bash
# Read-only snapshot for the handoff document.
cd "${BDC_ROOT}/experiments/claude_carla_overnight"
date
echo "--- ChatGPT queue still waiting?"; pgrep -af run_second_speed | cut -c1-90 || true
echo "--- trainings"
for t in second_twin_ll_s1 second_twin_ll_s2; do echo "$t: $(grep -E 'Train:' ${BDC_PROGRAM}/$t.log | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-100)"; done
echo "--- zips"; ls -la ${BDC_OUT}/bdc2026_trackB_submission_v*.zip | awk '{print $5, $9}'
echo "--- candidate dirs"; ls -d final/candidate_v*/ | tr '\n' ' '; echo
echo "--- checkpoints (tags)"; ls ${BDC_PROGRAM}/OpenPCDet/output${BDC_OUT}/'BIG DATA IEEE'/experiments/claude_carla_overnight/canon/*/ 2>/dev/null | grep -v -E '^$|:$' | tr '\n' ' ' | cut -c1-1500; echo
echo "--- datasets"; ls ${BDC_DATA} | tr '\n' ' '; echo
echo "--- key outputs"; ls coop/dev150/pred_v5_norule.json coop/dev350/pred_v5_norule.json refiner/*.log twin/run_twin_ll.log 2>&1 | tr '\n' ' '; echo
echo "--- OSC"; ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 ${OSC_USER}@ascend.osc.edu 'source /etc/profile 2>/dev/null; squeue -u ${OSC_USER} -o "%i %P %T %M %l %R"; squeue -j 8548278 --start -o "%i %S" | tail -n 1' 2>&1 | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-160
echo "--- Globus transfer"; ~/gcp/globusconnectpersonal -status 2>/dev/null | tr '\n' ' '; echo
