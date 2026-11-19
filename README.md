# IEEE BigData Cup 2026 — UrbanTwin Sim2Real LiDAR Challenge (V2X-Real Track)

**Site-Calibrated Synthetic Data and Multi-Detector Fusion for Roadside 3D Detection**
Arnav Kumar, Brijesh Soni — The Ohio State University (Codabench team `arnav007`)

| Final submission | Combined | 3D mAP@0.5 | Vehicle | Pedestrian | Truck | 3D mAP@0.7 | Realism |
|---|---|---|---|---|---|---|---|
| ID **1054222** (`v18s`) | **0.8947** | 0.6001 | 75.59 | 44.95 | 59.50 | 0.2646 | 0.9507 |

Ranked joint first on the official final leaderboard. The report is in [`report/`](report/).

All detectors are trained **from scratch on synthetic data only**. Real data is used only as described in [Compliance](#compliance).

---

## Repository layout

| Path | Contents |
|---|---|
| `submission/` | **The exact scored files of submission 1054222**: `predictions.json` and the 50 synthetic scans |
| `weights/` | **Trained weights of all 18 models in the final ensemble** (release assets) with configs, training sets and SHA-256 hashes |
| `generator_v2/` | **Generator V2 code** (byte-identical to the version that produced its training frames) and its OpenPCDet converter |
| `pipeline/env.sh` | **The one path configuration** (`BDC_ROOT`) used by every script |
| `ENVIRONMENT.md` | Exact software versions, including the OpenPCDet commit |
| `third_party/openpcdet_233f849.patch` | Our 4 small changes to OpenPCDet (apply before training) |
| `pipeline/final/reproduce_v18s.sh` | Rebuilds the final `predictions.json` on a CPU from the raw per-model predictions and checks it against `submission/` |
| `LICENSE` | MIT License |
| `report/` | Final report (PDF) and LaTeX source |
| `pipeline/carla_gen_v3.py`, `convert_carla_v3.py`, `build_mix.py` | CARLA roadside capture, conversion to OpenPCDet format, Generator V2 + CARLA training mix |
| `pipeline/canon/` | Ground-canonical datasets (`build_canon.py`) and all OpenPCDet detector configs (`*.yaml`) |
| `pipeline/twin/` | Site-calibrated twin: static map, range-image background, twin scene sets, submitted realism scans |
| `pipeline/osc/pvrcnn_train.sbatch` | OpenPCDet training job used on OSC (same command works locally) |
| `pipeline/codex_review/per_scan_export.py` | Per-scan ground levelling wrapper used for all test predictions |
| `pipeline/final/` | Inference, fusion (`fuse_test.py`), post-processing (`calib.py`) and the build of the final zip |
| `pipeline/coop/` | Development-set evaluation and ablations reported in the paper |
| `pipeline/LEARNINGS.md` | Detailed experiment notes |

---

## Requirements

- Linux (we used WSL2 Ubuntu), Python 3.10, CUDA 11.8, PyTorch with [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) built from source (`spconv`).
- CARLA 0.9.16 (only for generating the CARLA part of the synthetic data).
- The organizers' starting kit (`starting_kit/track_b`: scorer, 50 test scans, forbidden-frame list) and the V2X-Real infrastructure scans.
- Hardware used: one RTX 4070 laptop GPU and A100 40 GB jobs on OSC; about 110 + 47 GPU hours in total.

**Paths.** No script contains a machine-specific path. Everything is derived from **one configurable root, `BDC_ROOT`**, set in [`pipeline/env.sh`](pipeline/env.sh):

```bash
export BDC_ROOT=/path/to/workdir                                   # the only path to set (avoid spaces)
bash pipeline/setup_workdir.sh /path/to/starting_kit/track_b      # one-time: creates the expected layout via links
source pipeline/env.sh                                             # in every new shell
```

`env.sh` derives `BDC_DATA` (synthetic datasets), `BDC_PROGRAM` (contains `OpenPCDet/`), `BDC_PYTHON`, `BDC_OUT` (zips) and `BDC_REALISM_DEPS` from `BDC_ROOT`; each can be overridden. Shell scripts use `${BDC_ROOT}` etc., Python scripts `os.environ['BDC_ROOT']`.

### Quick check without a GPU (a few minutes)

The release also contains `raw_test_predictions.zip`: the per-model detections on the 50 test scans (per-scan levelled, six ground heights) from the 22 passes of the final ensemble. From them, fusion and post-processing rebuild the scored file exactly:

```bash
source pipeline/env.sh
curl -fLO https://github.com/Arnavvvkumar/IEEE-BigData-2026-Cup---V2X-Track/releases/download/v1.0/raw_test_predictions.zip
unzip -q raw_test_predictions.zip -d "$BDC_ROOT"
RAW="$BDC_ROOT/raw_test_predictions" bash pipeline/final/reproduce_v18s.sh
# -> reproduced {...} | identical to submission/predictions.json: True
```

The pedestrian static-structure map is read from `pipeline/final/static_map.npz` (voxel map and ground plane derived from the unlabelled development clouds), so no real data is needed for this check.

**Shortcut without retraining:** download the trained weights (`bash weights/download_weights.sh`) and start at Step 3. Install OpenPCDet as described in [ENVIRONMENT.md](ENVIRONMENT.md) (commit `233f849` plus our patch).

---

## Reproducing the final submission

### Step 1 — Synthetic training data

1. **CARLA scenes.** Start a CARLA server, then capture roadside clips with the sensor matched to the real one (2.87 m mount, ~6.5° tilt) and convert them:
   ```bash
   python pipeline/carla_gen_v3.py --help
   python pipeline/convert_carla_v3.py --help
   ```
2. **Canonical set** (`claude_canon_all_v1`, 10,800 frames): merge the Generator V2 frames with the CARLA frames (`build_mix.py`), then level every frame with the same ground fit used at test time:
   ```bash
   python pipeline/canon/build_canon.py --help
   ```
   The Generator V2 frames (`codex_openpcdet_custom_v1`, 5,000 + 500 val) are produced by [`generator_v2/`](generator_v2/) (procedural meshes ray cast with Open3D, no real data):
   ```bash
   python generator_v2/scripts/generate_synthetic_v2.py --config generator_v2/configs/generator_v2.yaml \
     --output "$BDC_DATA/claude_v2_scale5500_fix1" --train 5000 --validate 500 --seed 2026 --workers 8
   python generator_v2/prepare_openpcdet_data.py --out "$BDC_DATA/codex_openpcdet_custom_v1"
   ```
3. **Site-calibrated twin sets.** These need **unlabelled** real development clouds only (never the forbidden frames, never labels):
   ```bash
   python pipeline/twin/static_map.py      # static voxel map (used for pedestrian down-weighting)
   python pipeline/twin/range_bg.py        # range-image background model of the fixed sensor
   bash pipeline/twin/run_twin_full.sh     # claude_twin_v1 (5,000)  -> merged claude_canon_all_twin_v1
   bash pipeline/twin/run_twin_st.sh       # claude_twin_st_v1 (3,000, small trucks) -> claude_canon_all_twin_st_v1 (18,800)
   bash pipeline/twin/run_twin_pv.sh       # claude_twin_pv_v1 (5,000, size-varied pedestrians) -> claude_canon_all_twin_pv_v1
   ```
4. **Submitted realism scans** (`twin/final_synthetic_v3`, 50 scans, no-return points trimmed to the real fraction):
   ```bash
   python pipeline/twin/make_synthetic_v3.py
   ```

### Step 2 — Train the detectors

Every model is a standard OpenPCDet training run from scratch, for example:

```bash
cd "$BDC_PROGRAM/OpenPCDet/tools"
"$BDC_PYTHON" train.py --cfg_file "$BDC_ROOT/experiments/claude_carla_overnight/canon/pvrcnn_ped.yaml" --batch_size 4 --epochs 5 \
  --extra_tag pvrcnn_ped_P1 --set DATA_CONFIG.DATA_PATH "$BDC_DATA/claude_canon_all_twin_st_v1"
```

On a Slurm cluster, `pipeline/osc/pvrcnn_train.sbatch` runs the same command (set `TAG`, `CFG`, `DATA`, `EPOCHS`, `BS`). Random seeds are not fixed; the `_s1`, `_s2`, `_P1`… suffixes are repeated runs.

The models in the final ensemble (checkpoint epoch in the last column):

| Tag | Config (`pipeline/canon/`) | Dataset | Classes voted | Epoch |
|---|---|---|---|---|
| `vnx010_all_s1234` | `vnx010_all.yaml` | `claude_canon_all_v1` | pedestrian, vehicle | 10 |
| `vnx010_twin_s1` | `vnx010_all.yaml` | `claude_canon_all_twin_v1` | pedestrian, vehicle | 12 |
| `vnx010_all_rs2_amp` | `vnx010_all.yaml` | `claude_canon_all_v1` | pedestrian | 12 |
| `vnx010_pv_ft`, `vnx010_pv_ft2`, `vnx010_pv_ft4` | `vnx010_all.yaml` (fine-tuned from our own synthetic checkpoints) | `claude_canon_all_twin_pv_v1` | vehicle | 5, 5, 4 |
| `canon_all_s1234` | `second_canon_all_v1.yaml` | `claude_canon_all_v1` | vehicle, truck | 12 |
| `second_twin_s1`, `second_twin_s2` | `second_canon_all_v1.yaml` | `claude_canon_all_twin_v1` | vehicle, truck | 12 |
| `canon_relabel_s1234` | `second_canon_v1.yaml` | `claude_canon_relabel_mix6000_v1` | vehicle, truck | 12 |
| `second_twin_st_s1` | `second_canon_all_v1.yaml` | `claude_canon_all_twin_st_v1` | truck | 12 |
| `second_twin_pv_s1` | `second_canon_all_v1.yaml` | `claude_canon_all_twin_pv_v1` | truck | 14 |
| `pvrcnn_st_osc` | `pvrcnn_all.yaml` | `claude_canon_all_twin_st_v1` | pedestrian, vehicle, truck | 6 |
| `pvrcnn_st_osc_s2` | `pvrcnn_all.yaml` | `claude_canon_all_twin_st_v1` | pedestrian, vehicle | 4 |
| `pvrcnn_ped_P1`, `_P2`, `_P3` | `pvrcnn_ped.yaml` | `claude_canon_all_twin_st_v1` | pedestrian (each counted 3×) | 5 |
| `pvrcnn_ped_P4` | `pvrcnn_ped_aug.yaml` | `claude_canon_all_twin_st_v1` | pedestrian | 5 |

The long-low truck model (`second_twin_ll_*`) is **not** part of the final ensemble.

### Step 3 — Per-scan levelled inference on the 50 test scans

Each test scan is levelled with the ground plane fitted **from its own points only** (`PLANE=scan`) and run at six ground heights (−3.15 … −2.775 m):

```bash
bash pipeline/final/reexport_perscan_b1.sh        # SECOND, VoxelNeXt, PV-RCNN voters + VoxelNeXt TTA -> final/test_allpred_ps1/
bash pipeline/final/_reexport_p27.sh              # pvrcnn_ped_P3
bash pipeline/final/_reexport_pvrcnn_ped_P4.sh    # pvrcnn_ped_P4
bash pipeline/final/p30_tta_P123.sh               # PV-RCNN pedestrian TTA passes (P1, P3)
```

`codex_review/per_scan_export.py` wraps `final/infer_test.py` / `final/infer_tta.py` and enforces the per-scan plane; the TTA passes use small shifts and rotations (`--variants 0.05,0.05,0 0.05,0,7 0,0.05,-7 0.025,0.075,14 0.075,0.025,-14`) at four ground heights.

### Step 4 — Fusion, post-processing and the final zip

The final zip `v18s` takes its **vehicle and truck boxes from build v13t** and its **pedestrian boxes from build v17**:

```bash
bash pipeline/final/run_v17.sh   # v17: pedestrian voters + averaged pedestrian heading + <5-point pedestrian rule
bash pipeline/final/run_v18.sh   # v13t (with the <30-point truck rule), splice with v17 -> bdc2026_trackB_submission_v18s.zip
```

Inside these builds (`build_candidate_v7.sh`, `fuse_test.py`, `calib.py`):

1. **Fusion** over all (model, height) passes: BEV-centre clustering, score-weighted mean box, score = sum of member scores / number of passes; circular-mean heading for vehicles and pedestrians (`FUSE_YAW_CIRC=vehicle,pedestrian`); one vote per pass for vehicles (`FUSE_ONE_VOTE=vehicle`).
2. **Pedestrian static down-weight**: score × (1 − 0.8 × static fraction).
3. `calib.calibrate` (box-offset calibration) → `truck_rule` (a 0.5, base 0.1) → `truck_dedupe` → `truck_min_points` (< 30 points: × 0.25) → `truck_boost_hedge` → `ped_min_points` (< 5 points: × 0.25) → `hedge_pedestrians`.
4. The 50 synthetic scans from `twin/final_synthetic_v3` and `declaration.txt` are added; `splice_ped.py` combines the two builds.

### Step 5 — Validate the zip

```bash
bash pipeline/final/kit_validate.sh bdc2026_trackB_submission_v18s.zip   # organizers' scorer, expects "valid_submission": 1
```

Expected contents: 50 frames with 21,892 vehicle, 33,173 pedestrian and 17,783 truck boxes, and 50 synthetic scans.

---

## Development evaluation

`pipeline/coop/` contains the evaluation on the two screened development sets (150 and 350 frames), always reported on the full set and its two alternating halves. `coop/eval150.py` fuses cached predictions with a given pass list and scores them with the organizers' `detection_eval.py`; the ablation scripts behind Table IV of the report are in the same folder.

---

## Not included

- **Real data**: V2X-Real scans and labels and the organizers' forbidden-frame list (their licence).
- **Our synthetic training sets** (tens of GB); they are regenerated by the scripts in Step 1 and `generator_v2/`.
- **Intermediate predictions and logs.** The final weights (`weights/`) and the final submission files (`submission/`) are included.

---

## Compliance

- All detectors were trained from scratch on synthetic data only; no real labels or real object points were used for training.
- Unlabelled real development scans were used only to calibrate the synthetic pipeline (range image, no-return statistics, range histogram of the submitted scans) and to build the static-structure map.
- Labelled development scans were used only for validation, model selection, box-size statistics and post-processing constants.
- No frame from the organizers' forbidden-frame list was used at any stage (`pipeline/final/check_forbidden.py`).
- Each test scan is processed independently; no pseudo-labels, self-training or tuning on test scans.

## License

[MIT](LICENSE). It covers the code, configuration files, the released model weights and the submission files in this repository. The detectors are built with OpenPCDet (Apache License 2.0), which is not included here; the V2X-Real data and the organizers' starting kit are not distributed and remain under their own terms.

## Citation

If you use this code, please cite our report (see `report/`) and the challenge papers by Shahbaz and Agarwal (IEEE T-ITS 2026; IEEE OJ-ITS 2026).
