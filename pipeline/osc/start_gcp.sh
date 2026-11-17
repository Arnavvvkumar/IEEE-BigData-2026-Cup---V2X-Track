#!/bin/bash
# Share ONLY the dataset folder (read-only) and start Globus Connect Personal in the background.
mkdir -p ~/.globusonline/lta
echo "$HOME/projects/bdc2026-track-b/data/,0,0" > ~/.globusonline/lta/config-paths
cat ~/.globusonline/lta/config-paths
cd ~/gcp
setsid nohup ./globusconnectpersonal -start > ~/gcp_start.log 2>&1 < /dev/null &
sleep 25
./globusconnectpersonal -status
ls ~/projects/bdc2026-track-b/data | head -n 20
du -sh ~/projects/bdc2026-track-b/data/claude_twin_st_v1 ~/projects/bdc2026-track-b/data/claude_canon_all_v1 ~/projects/bdc2026-track-b/data/claude_twin_v1 2>/dev/null
