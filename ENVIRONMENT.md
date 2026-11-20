# Environment

The exact software used for the final submission (1054222).

## Detection (training and inference)

| Component | Version |
|---|---|
| OS | Ubuntu (WSL2 on Windows 11) for local runs; Ohio Supercomputer Center Ascend (Linux) for A100 jobs |
| Python | 3.10.22 |
| PyTorch / torchvision | 2.2.2+cu118 / 0.17.2+cu118 |
| CUDA toolkit | 11.8 |
| spconv | 2.3.6 (`spconv-cu118`) |
| OpenPCDet | commit [`233f849`](https://github.com/open-mmlab/OpenPCDet/commit/233f849829b6ac19afb8af8837a0246890908755) **plus** [`third_party/openpcdet_233f849.patch`](third_party/openpcdet_233f849.patch) |
| numpy / scipy | 1.23.5 / 1.10.1 |
| numba / llvmlite | 0.58.1 / 0.41.1 |
| easydict, PyYAML, tensorboardX, SharedArray | 1.13, 6.0.2, 2.6.2.2, 3.2.4 |
| scikit-image / scikit-learn / shapely | 0.21.0 / 1.7.2 / 2.0.6 |
| GPUs | NVIDIA GeForce RTX 4070 Laptop (driver 610.74) and NVIDIA A100 40 GB |

The checkpoints in release `v1.0` report `pcdet+0.6.0` as their training version.

### OpenPCDet patch (5 small changes) and base configs

```bash
source pipeline/env.sh
git clone https://github.com/open-mmlab/OpenPCDet.git "$BDC_PROGRAM/OpenPCDet"
cd "$BDC_PROGRAM/OpenPCDet" && git checkout 233f849
git apply /path/to/this/repo/third_party/openpcdet_233f849.patch
python setup.py develop
# the two VoxelNeXt base configs that canon/make_cfg.py derives the final vnx010_*/vnx020_* configs from:
cp /path/to/this/repo/pipeline/base_configs/*.yaml tools/cfgs/custom_models/
```

1. `pcdet/datasets/dataset.py`: frames without objects are **kept** during training as negative examples (stock OpenPCDet re-samples them). This affects training.
2. `pcdet/datasets/custom/custom_dataset.py`: empty label files load as a `(0, 7)` box array instead of failing.
3. `tools/train.py`: adds `--skip_builtin_eval` (used by our training commands; evaluation is done with the challenge scorer) and changes the fixed seed from 666 to 1234. The seed applies only to runs started with `--fix_random_seed`, i.e. the models tagged `_s1234`; all other runs are unseeded, so retraining them gives similar but not bit-identical weights (use the released weights for exact results).
4. `pcdet/datasets/__init__.py`: the Argoverse 2 dataset is imported only if its package (`av2`) is installed.
5. `pcdet/config.py`: `${VAR}` in string values of a config is expanded when the yaml is loaded, so the configs in `pipeline/` use `DATA_PATH: ${BDC_DATA}/<dataset>` (set by `pipeline/env.sh`). Without this change, pass the dataset path explicitly (`--set DATA_CONFIG.DATA_PATH ...`, or `--data` for the inference scripts).

## Synthetic data generation

| Component | Version |
|---|---|
| Generator V2 | Python 3.10.22, numpy 1.23.5, open3d 0.18.0, PyYAML 6.0.2 (CPU only) |
| CARLA scenes | CARLA 0.9.16 (Windows build, `-nullrhi`, CPU) with its Python client |
| Site-calibrated twin | Python 3.10, numpy, scipy (CPU only) |

## Reproducing without a GPU

`pipeline/final/reproduce_v18s.sh` rebuilds the final `predictions.json` from the raw per-model test predictions
(release asset `raw_test_predictions.zip`) in a few minutes on a CPU, and checks it against `submission/predictions.json`.
