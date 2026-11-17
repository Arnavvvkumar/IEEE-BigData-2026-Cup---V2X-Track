#!/bin/bash
# Read-only: data-upload size, env-build log tail, smoke queue.
set -euo pipefail
SSHCMD="ssh -i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40"
echo "=== local upload ==="
pgrep -af 'upload_rsync|rsync -aL' || echo "uploader not running"
echo "uploader_count $(pgrep -fc upload_rsync.sh || true)"
tail -n 4 "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc/upload_rsync.log"
echo "=== remote ==="
$SSHCMD <osc_user>@ascend.osc.edu 'bash -s' << 'REMOTE' 2>/dev/null | grep -v -E '^\*|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials|^$'
echo "data:"
du -sm /fs/scratch/<osc_project>/bdc/data/* 2>/dev/null || true
echo "env:"
if [ -x /fs/scratch/<osc_project>/bdc/env/bin/python ]; then echo venv_present; else echo venv_absent; fi
echo "build procs:"
pgrep -af 'remote_build_env|setup.py' || echo none
echo "log tail:"
tail -n 30 /fs/scratch/<osc_project>/bdc/logs/build_env.log 2>/dev/null || echo no_build_log
echo "squeue:"
squeue -u <osc_user> || true
REMOTE
