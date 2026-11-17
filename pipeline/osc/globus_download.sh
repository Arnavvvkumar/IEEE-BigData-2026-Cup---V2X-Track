#!/bin/bash
set -euo pipefail
cd "$HOME"
if [ -d "$HOME/.globusonline/lta" ]; then
  echo "EXISTING_CONFIG"
  ls -la "$HOME/.globusonline/lta" || true
fi
if ls -d "$HOME"/globusconnectpersonal-3.* >/dev/null 2>&1; then
  echo "ALREADY_EXTRACTED"
  ls -d "$HOME"/globusconnectpersonal-3.*
  exit 0
fi
wget -O "$HOME/globusconnectpersonal-latest.tgz" https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
tar xzf "$HOME/globusconnectpersonal-latest.tgz"
ls -d "$HOME"/globusconnectpersonal-3.*
echo "DOWNLOAD_OK"
