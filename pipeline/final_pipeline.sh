#!/bin/bash
# Post-capture CPU pipeline: refinement table -> conversion -> CARLA audit -> tests -> mix build -> mix audit -> Codex verifier.
# Does NOT write SECOND_MIX_READY.json (done by hand after reviewing every audit).
set -euo pipefail
H="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
D=~/projects/bdc2026-track-b/data
E=~/projects/bdc2026-track-b/.env/bin/python
CONV=$D/claude_carla64_town02_v1
MIX=$D/claude_v2carla_mix6000_v1
cd "$H"
run() { nice -n 10 taskset -c 6-13 "$@"; }
echo "== refinement"; run $E refine_boxes.py --native native_v1 dryrun_nullrhi_nowalkers --out bbox_refinement.json
echo "== tests"; run $E -m unittest test_convert_carla 2>&1 | tail -3
echo "== convert"; run $E convert_carla.py --native native_v1 --out $CONV
echo "== carla audit"; run $E audit_carla.py --native native_v1 --converted $CONV --out audit_carla_v1.json
echo "== walkers"; run $E tools/walker_counts.py native_v1 $CONV
echo "== mix"; bash tools/mix_test.sh $CONV $MIX
sha256sum bbox_refinement.json convert_carla.py carla_gen.py audit_carla.py build_mix.py audit_mix.py refine_boxes.py $CONV/frames.json $CONV/provenance.json $MIX/manifest.json $MIX/custom_infos_train.pkl $MIX/ImageSets/train.txt $MIX/ImageSets/val.txt > final_hashes.txt
cat final_hashes.txt
