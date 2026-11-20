#!/bin/bash
# Path configuration for every script in this repository.
#   source pipeline/env.sh
# The one path you need to set is BDC_ROOT (a working directory; avoid spaces in it).
# Every other location defaults to a folder inside BDC_ROOT and can be overridden individually.

export BDC_ROOT="${BDC_ROOT:-$HOME/bdc2026}"

export BDC_DATA="${BDC_DATA:-$BDC_ROOT/data}"                      # synthetic training sets (OpenPCDet custom format)
export BDC_PROGRAM="${BDC_PROGRAM:-$BDC_ROOT/detector-program}"    # contains OpenPCDet/ (and model output/ dirs)
export BDC_PYTHON="${BDC_PYTHON:-python3}"                         # Python with OpenPCDet, torch, numpy, open3d
export BDC_OUT="${BDC_OUT:-$BDC_ROOT/out}"                         # submission zips and staging files
export BDC_REALISM_DEPS="${BDC_REALISM_DEPS:-$BDC_ROOT/realism_deps}"  # optional: extra packages for the realism metrics (POT)

# Only for the scripts in pipeline/osc (Ohio Supercomputer Center): your OSC login and project account
export OSC_USER="${OSC_USER:-your_osc_username}"
export OSC_PROJECT="${OSC_PROJECT:-your_osc_project}"

# Expected layout under BDC_ROOT (created by pipeline/setup_workdir.sh):
#   $BDC_ROOT/experiments/claude_carla_overnight  -> this repository's pipeline/ folder
#   $BDC_ROOT/starting_kit/track_b                -> the organizers' starting kit (scorer, 50 test scans)
#   $BDC_ROOT/scripts, $BDC_ROOT/configs          -> this repository's generator_v2/scripts and generator_v2/configs
mkdir -p "$BDC_OUT" 2>/dev/null || true
