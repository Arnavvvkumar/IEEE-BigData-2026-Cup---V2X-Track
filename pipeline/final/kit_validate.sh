#!/bin/bash
# Run the organizers' own local_eval.py (same scoring program as the server) on a submission zip. The kit's dummy reference expects 5 synthetic
# scans, the real challenge 50, so a temporary reference is built: dummy ground truth (zeros: AP is meaningless) + n_realism_frames = 50 +
# 50 unlabelled real DEVELOPMENT clouds as the realism reference. What matters: "valid_submission": 1. Nothing is uploaded.
# usage: kit_validate.sh <zip name in Downloads>
zip="${BDC_OUT}/$1"
W="${BDC_ROOT}"
kit="$W/starting_kit/track_b"
tmp=$(mktemp -d)
cp -r "$kit/reference_data_dummy" "$tmp/ref"
rm -f "$tmp/ref/realism_reference/"*
ls "$W/experiments/real_reference_audit/real_dev_cw1/detection_test_frames/"*.bin | head -n 50 | while read f; do cp "$f" "$tmp/ref/realism_reference/"; done
sed -i 's/"n_realism_frames": 5/"n_realism_frames": 50/' "$tmp/ref/config.json"
echo "reference frames: $(ls $tmp/ref/realism_reference | wc -l) | config: $(grep n_realism_frames $tmp/ref/config.json)"
cp "$zip" "$tmp/sub.zip"
export PYTHONPATH="${BDC_REALISM_DEPS}"
timeout 900 "${BDC_PYTHON}" "$kit/local_eval.py" "$tmp/sub.zip" "$tmp/ref" 2>&1 | grep -v -E 'Warning|warn' | tail -n 12 | cut -c1-200
echo "== scores.json (AP and realism are NOT meaningful here: dummy ground truth / dev reference)"; head -c 700 "$kit/scores.json"; echo
