#!/bin/bash
# Read-only snapshot of everything running, for the handoff document.
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
echo "== date"; date
echo "== training processes (extra_tag)"; pgrep -fa 'train.py' | grep -o 'extra_tag [a-z0-9_]*' | sort | uniq -c
echo "== latest training log line per run"
for t in vnx010_all_rs3_amp second_twin_s2 vnx010_twin_s2; do echo "-- $t"; tail -c 600 ~/projects/detector-program/$t.log 2>/dev/null | tr '\r' '\n' | tail -n 1 | cut -c1-170; done
echo "== checkpoints"
for t in vnx010_all_rs3_amp second_twin_s2 vnx010_twin_s2; do f=$(find ~/projects/detector-program/OpenPCDet/output -path "*/$t/ckpt/checkpoint_epoch_*.pth" 2>/dev/null | sed 's/.*epoch_//' | sort -n | tail -n 1); echo "$t latest epoch file: ${f:-none}"; done
echo "== scored results for new runs"; grep -E 'vnx010_all_rs3_amp|second_twin_s2|vnx010_twin_s2' "$base/canon/results/summary.txt" | head -n 3 | cut -c1-150
echo "== OSC upload"; tail -n 4 "$base/osc/upload_rsync.log" 2>/dev/null | cut -c1-160
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=20 <osc_user>@ascend.osc.edu 'du -sm /fs/scratch/<osc_project>/bdc/data/* /fs/scratch/<osc_project>/bdc/code 2>/dev/null; ls /fs/scratch/<osc_project>/bdc/cfg 2>/dev/null | head -n 20' 2>/dev/null | grep -v -E '^\*|^\s*$|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials'
echo "== background rsync/push procs"; pgrep -fa 'rsync|push_code|upload_rsync' | cut -c1-100
echo "== free mem"; free -m | sed -n 2p
echo "== eval summary (150 frames)"; cut -c1-220 "$base/coop/dev150/eval_summary.txt" 2>/dev/null
