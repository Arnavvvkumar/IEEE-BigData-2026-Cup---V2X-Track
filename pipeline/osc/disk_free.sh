#!/bin/bash
set -euo pipefail
df -h "$HOME" /tmp
echo "---"
du -sh "${BDC_DATA}/claude_twin_v1" "${BDC_DATA}/claude_canon_all_v1"
