#!/bin/bash
# Put the user's filled-in declaration into the candidate folder, rebuild the zip, and run the scorer's declaration check on it.
# usage: finish_zip.sh <candidate dir name> <zip name in Downloads>
set -euo pipefail
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final"
dl="/mnt/c/Users/arnav/Downloads"
P="$HOME/projects/bdc2026-track-b/.env/bin/python"
cd "$here"
cp "$dl/declaration.txt" "$1/declaration.txt"
rm -f "$dl/$2"
"$P" make_zip.py "$1" "$dl/$2"
unzip -p "$dl/$2" declaration.txt | grep -i -c 'I confirm that I did not train on any frame listed in'
unzip -l "$dl/$2" | grep -v 'synthetic/00' | tail -n 8
