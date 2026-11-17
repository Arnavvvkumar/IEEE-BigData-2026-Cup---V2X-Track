# IEEE BigData Cup 2026 — UrbanTwin Sim2Real LiDAR Challenge (V2X-Real Track)

Team **arnav007 / Fusion Team** — Arnav Kumar, Brijesh Soni (The Ohio State University).
Final submission: Codabench ID **1054222**, combined score **0.8947**
(vehicle 75.59, pedestrian 44.95, truck 59.50 AP @ IoU 0.5 R40; realism 0.9507).

## Contents

| Path | What it is |
|---|---|
| `report/IEEE_BigData_Cup_Report_V2X_Final.pdf` | Final report (as emailed to the organizers) |
| `report/source/` | LaTeX source of the report |
| `pipeline/` | All code and configs used in the project (copied from the working tree) |
| `pipeline/LEARNINGS.md` | Detailed log of every experiment from v7 onward: what worked, what failed, and why |

### Key parts of `pipeline/`

| Path | Role |
|---|---|
| `carla_gen*.py`, `convert_carla*.py`, `build_mix.py` | CARLA scene generation and conversion to the challenge format |
| `twin/` | Site-calibrated background (range image, no-return beams) and twin scene generation; realism tuning of the submitted scans |
| `canon/` | OpenPCDet detector configs (SECOND, VoxelNeXt, PV-RCNN, specialists) and local training scripts |
| `osc/` | Training jobs on the Ohio Supercomputer Center (account/user replaced by `<osc_project>` / `<osc_user>`) |
| `final/infer_test.py`, `codex_review/per_scan_export.py` | Levelled multi-height inference; per-scan ground levelling (`PLANE=scan`) used for the final test predictions |
| `final/fuse_test.py` | BEV cluster fusion across (model, height) passes, circular-mean heading, one-vote-per-pass, pedestrian static down-weight |
| `final/calib.py` | Box calibration, truck rule and de-duplication, truck size hedge, low-point pedestrian/truck rules, pedestrian hedging |
| `final/build_candidate_v7.sh`, `final/run_v18.sh`, `final/splice_ped.py` | Build of the final submission zip (v18s) |
| `coop/` | Development-set evaluation and every ablation reported in the paper |

## Final submission recipe (v18s)

1. Per-scan levelled inference of every ensemble member at six ground heights on each test scan (`PLANE=scan`; no other scan used).
2. Fusion with `FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle`; the long-low truck voter (`second_twin_ll_*`) is excluded.
3. `calibrate` → `truck_rule` (a 0.5, base 0.1) → `truck_dedupe` → `truck_min_points` (<30 pts ×0.25) → `truck_boost_hedge` → `ped_min_points` (<5 pts ×0.25) → `hedge_pedestrians`.
4. Vehicle and truck boxes from build v13t, pedestrian boxes from build v17 (`splice_ped.py`); synthetic realism scans from `candidate_v3`.

## Not included

- **No data**: V2X-Real scans and labels (licence), the organizers' forbidden-frame list, our synthetic datasets, and the submitted scans are not redistributed.
- **No model weights, predictions, logs or submission zips** (tens of GB).
- Scripts contain absolute paths from the authors' machines; adjust them to your environment.

## Compliance

All detectors were trained from scratch on synthetic data only. The 100 forbidden frames were excluded from every stage. Unlabelled real development scans were used only for site calibration, the static-structure map and realism statistics; labelled development scans only for validation, model selection and post-processing constants. Each test scan is processed independently, with no tuning on test scans, no pseudo-labels and no self-training. See Section II-D of the report.
