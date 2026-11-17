#!/bin/bash
# Candidate v10 = v9 + P1 (pedestrian-only PV-RCNN, epoch 5) counted 3 times as a pedestrian voter + one confidence vote per pass for vehicles
# (Codex's finding; FUSE_ONE_VOTE=vehicle). Everything else as v9: job 1 epoch 6 (ped, veh, truck), job 2 epoch 4 (ped, veh), truck size hedge
# (no boost), pedestrian hedging, zero-trimmed synthetic scans (twin/final_synthetic_v3). Writes candidate_v12 and the v10 zip only.
set -euo pipefail
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final"
dl="/mnt/c/Users/arnav/Downloads"
tag="v12-final-2026-11-14"
cp "$dl/declaration.txt" "$here/_declaration_backup_12.txt"
sed -i "s/^Date (YYYY-MM-DD):.*/Date (YYYY-MM-DD):      2026-11-14/" "$dl/declaration.txt"; sed -i "s/^Submission tag:.*/Submission tag:         $tag/" "$dl/declaration.txt"
P1=pvrcnn_ped_P1_ep5:pedestrian
export FUSE_ONE_VOTE=vehicle; export TESTPRED="$here/test_allpred_ps1" DROP=second_twin_ll_s1_ep12
EXTRA="pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle $P1 $P1 $P1 pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian" TRUCKPP=1 TRUCKBOOST=1.0 \
  SYNDIR="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/twin/final_synthetic_v4/synthetic" \
  bash "$here/build_candidate_v7.sh" candidate_v12 bdc2026_trackB_submission_v12.zip 1
cp "$here/_declaration_backup_12.txt" "$dl/declaration.txt"   # restore the resting declaration file
echo V12_BUILT
