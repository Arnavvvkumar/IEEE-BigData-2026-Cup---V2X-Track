#!/bin/bash
set -euo pipefail
df -h "$HOME" /tmp
echo "---"
du -sh "$HOME/projects/bdc2026-track-b/data/claude_twin_v1" "$HOME/projects/bdc2026-track-b/data/claude_canon_all_v1"
