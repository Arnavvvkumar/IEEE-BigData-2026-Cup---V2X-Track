# Generator V2 (procedural synthetic roadside LiDAR)

Generator V2 produced the 5,000 training + 500 validation frames that form part of the canonical training set.
It builds procedural meshes (cars, trucks, buses, pedestrians, buildings, poles, trees, curbs) on sampled road layouts and
ray casts them on the CPU with Open3D. It uses no real data, no external assets and no real-data statistics.

`scripts/` and `configs/` are **byte-identical** to the code recorded in the generated dataset's `provenance.json`
(the SHA-256 of every file and of the config matches).

```bash
source pipeline/env.sh
# 1) generate the frames (about 4 minutes on 8 CPU cores)
python generator_v2/scripts/generate_synthetic_v2.py --config generator_v2/configs/generator_v2.yaml \
  --output "$BDC_DATA/claude_v2_scale5500_fix1" --train 5000 --validate 500 --seed 2026 --workers 8
# 2) convert to the OpenPCDet custom format used for training (codex_openpcdet_custom_v1)
python generator_v2/prepare_openpcdet_data.py --out "$BDC_DATA/codex_openpcdet_custom_v1"
```

`prepare_openpcdet_data.py` is the converter we used, with its input path made configurable (`$BDC_DATA`).
Requirements: Python 3.10, numpy 1.23, open3d 0.18, pyyaml 6.
