#!/bin/bash
# Share ONLY the dataset folder (read-only) and start Globus Connect Personal in the background.
mkdir -p ~/.globusonline/lta
echo "${BDC_DATA}/,0,0" > ~/.globusonline/lta/config-paths
cat ~/.globusonline/lta/config-paths
cd ~/gcp
setsid nohup ./globusconnectpersonal -start > ~/gcp_start.log 2>&1 < /dev/null &
sleep 25
./globusconnectpersonal -status
ls ${BDC_DATA} | head -n 20
du -sh ${BDC_DATA}/claude_twin_st_v1 ${BDC_DATA}/claude_canon_all_v1 ${BDC_DATA}/claude_twin_v1 2>/dev/null
