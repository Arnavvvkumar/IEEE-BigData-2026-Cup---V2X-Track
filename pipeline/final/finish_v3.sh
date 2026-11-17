#!/bin/bash
# After dev350 predictions exist: score v2 (no truck rule) and v3. Package only if v3
# is strictly better on every class on both halves. Does not submit the zip.
set -euo pipefail
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
here="$root/experiments/claude_carla_overnight/final"
coop="$root/experiments/claude_carla_overnight/coop"
P="$HOME/projects/detector-program/env310/bin/python"
cd "$coop"
V2="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck"
V3="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian"
need() {
  [ -f "$coop/dev350/allpred/${1}_allpred.json" ] || { echo "missing dev350 allpred $1"; exit 1; }
}
for t in vnx010_all_s1234_ep10 vnx010_all_rs2_amp_ep12 vnx010_twin_s1_ep12 canon_all_s1234_ep12 second_twin_s1_ep12 canon_relabel_s1234_ep12 vnx010_twin_s1_tta_ep12 vnx010_all_s1234_tta_ep10; do
  need "$t"
done
[ -f "$here/dev150_v3_ok" ] || { echo "dev150 pipeline not confirmed"; exit 1; }
# shellcheck disable=SC2086
"$P" eval150.py dev350 v2_norule $V2 norule
# shellcheck disable=SC2086
"$P" eval150.py dev350 v3_ttaBothPed $V3
"$P" - << 'PY'
import json
from pathlib import Path
summary = Path("/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop/dev350/eval_summary.txt").read_text().strip().splitlines()
rows = {}
for line in summary:
    name = line.split(" | ", 1)[0]
    if name in ("v2_norule", "v3_ttaBothPed"):
        rows[name] = json.loads(line.rsplit(" | ", 1)[-1])
v2, v3 = rows["v2_norule"], rows["v3_ttaBothPed"]
print("class all_v2 all_v3 A_v2 A_v3 B_v2 B_v3")
ok = True
for c in ("vehicle", "pedestrian", "truck"):
    print(c, v2[c][0], v3[c][0], v2[c][1], v3[c][1], v2[c][2], v3[c][2])
    if not (v3[c][1] > v2[c][1] and v3[c][2] > v2[c][2]):
        ok = False
        print("FAIL", c)
print("mAP", v2["mAP"], v3["mAP"])
Path("/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final/dev350_v3_ok").write_text("ok\n" if ok else "fail\n")
raise SystemExit(0 if ok else 2)
PY
echo "dev350 v3 beats v2 on both halves"
bash "$here/build_candidate_v3.sh"
bash "$here/finish_zip.sh" candidate_v3 bdc2026_trackB_submission_v3.zip
"$P" "$here/verify_zip.py" "/mnt/c/Users/arnav/Downloads/bdc2026_trackB_submission_v3.zip"
echo V3_PACKAGED
