#!/bin/bash
# Which synthetic folder did the submitted v9 zip use?
rm -rf /tmp/v9s && mkdir /tmp/v9s && cd /tmp/v9s
python3 -c "import zipfile; z=zipfile.ZipFile('/mnt/c/Users/arnav/Downloads/bdc2026_trackB_submission_v9.zip'); [z.extract(n) for n in z.namelist() if n.startswith('synthetic/')]"
T="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/twin"
ls synthetic | wc -l
for s in final_synthetic_v1 final_synthetic_v3 final_synthetic_v4; do
  echo "$s differing files: $(diff -rq synthetic "$T/$s/synthetic" 2>&1 | wc -l)"
done
