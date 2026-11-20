#!/bin/bash
# Show why the remote file list failed. Does not start another upload.
set -uo pipefail
echo "ssh test $(date -Is)"
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=25 ${OSC_USER}@ascend.osc.edu 'echo SSH_OK; ls -ld /fs/scratch/${OSC_PROJECT}/bdc/data/claude_twin_v1; find /fs/scratch/${OSC_PROJECT}/bdc/data/claude_twin_v1 -maxdepth 1 -type d' \
  2>&1 | grep -v -E '^\*|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials|^$'
echo "exit ${PIPESTATUS[0]}"
