#!/bin/bash
# One GPU inference at a time. Leaves vnx010_twin_s2 training alone.
# Test TTA, then dev350 six-height members, then the two dev350 TTA files.
set -uo pipefail
root="${BDC_ROOT}"
here="$root/experiments/claude_carla_overnight/final"
coop="$root/experiments/claude_carla_overnight/coop"
test_scans="$root/starting_kit/track_b/detection_test_frames"
test_out="$here/test_allpred"
dev_scans="$coop/dev350/scans"
dev_out="$coop/dev350/allpred"
P="${BDC_PYTHON}"
log="$here/queue_v3_infer.log"
exec >> "$log" 2>&1
echo "QUEUE_START $(date -Is)"
"$P" "$here/check_forbidden.py" "$coop/dev350" || { echo "dev350 forbidden overlap"; exit 1; }

ok_json() {
  "$P" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert len(d)>=int(sys.argv[2])' "$1" "$2"
}

run_tta() {
  # tag cfg dataset epoch scans outdir min_passes
  local out="$6/${1}_tta_ep${4}_allpred.json"
  local try
  for try in 1 2 3 4 5 6; do
    if [ -f "$out" ] && ok_json "$out" "$7"; then
      echo "have $out"
      return 0
    fi
    rm -f "$out"
    echo "tta try $try $1 $(date -Is)"
    bash "$here/run_tta_one.sh" "$1" "$2" "$3" "$4" "$5" "$6" || true
    if [ -f "$out" ] && ok_json "$out" "$7"; then
      echo "TTA_OK $1"
      return 0
    fi
    echo "tta not ready $1 try $try; sleep 120"
    sleep 120
  done
  echo "TTA_GAVE_UP $1"
  return 1
}

echo "=== test tta twin $(date -Is)"
run_tta vnx010_twin_s1 vnx010_all.yaml claude_canon_all_twin_v1 12 "$test_scans" "$test_out" 20 || exit 1
sleep 5
echo "=== test tta all $(date -Is)"
run_tta vnx010_all_s1234 vnx010_all.yaml claude_canon_all_v1 10 "$test_scans" "$test_out" 20 || exit 1
sleep 5
echo "=== dev350 six-height $(date -Is)"
bash "$here/run_infer_dir.sh" "$dev_scans" "$dev_out"
sleep 5
echo "=== dev350 tta twin $(date -Is)"
run_tta vnx010_twin_s1 vnx010_all.yaml claude_canon_all_twin_v1 12 "$dev_scans" "$dev_out" 20 || exit 1
sleep 5
echo "=== dev350 tta all $(date -Is)"
run_tta vnx010_all_s1234 vnx010_all.yaml claude_canon_all_v1 10 "$dev_scans" "$dev_out" 20 || exit 1
echo "QUEUE_INFER_DONE $(date -Is)"
bash "$here/finish_v3.sh"
echo "QUEUE_ALL_DONE $(date -Is)"
