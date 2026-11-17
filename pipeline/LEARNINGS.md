# LEARNINGS — IEEE BigData Cup 2026, UrbanTwin Sim2Real, Track B (V2X-Real)

Everything we learned, from the foundations through v7 and on to v17s: what worked, what did not, what wasted time, and why.
Written Nov 15 2026 (about 02:25 EST), deadline Nov 15 23:59 EST. Account: arnav007 (Codabench competition 16685).
Work root: `C:\Users\arnav\Downloads\BIG DATA IEEE\experiments\claude_carla_overnight\` (WSL: `/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/...`).

How to read it: section 1 is the cheat-sheet, section 2 the official scores, sections 4-7 the experiment log, section 8 the mistakes, section 9 the general lessons.
Numbers marked (est.) are estimates, not measurements.

---

## 1. Cheat-sheet (the 12 things that matter most)

1. **Score = 0.6 * min(mAP/0.7, 1) + 0.4 * realism.** +1 AP in one class (of 3) = +0.0029. +0.01 realism = +0.004.
2. **Ground levelling was the biggest single win.** Real scans are tilted about 6.5 deg; levelling each scan doubled real-dev mAP (15.8 -> 37) with no retraining.
3. **Fusion rewards agreement.** Score = (sum of member scores) / (number of passes) beat every alternative (max, mean, geometric) by 9-27 AP for trucks.
4. **Pedestrian hedging** (also emitting size/shift variants at lower scores) gave +4.5 AP on the real test. It works because the scorer has no cap on predictions and a variant ranks below its own parent.
5. **Averaged (circular-mean) heading** in fusion: vehicles +4.5 AP (test +5.2); pedestrians later +0.2 to +0.7 AP.
6. **Pedestrian dev AP predicts test AP about 1:1** (v9 -> v12: dev +1.7, test +2.17; v12 -> v15s: dev +0.6, test +0.67). **Vehicle and truck dev AP do not predict test.** Realism dev does not predict test at all (it is biased, not just noisy).
7. **Dev truck AP is inflated** by about 13 parked objects repeated in every frame (garbage truck "TrashCan", buses). Dev 88 vs test 59.
8. **More models of a type we already have add about nothing** to pedestrians once the ensemble is big (P4/P5/P6 PV-RCNN, Voxel R-CNN VR1/VR2: all <= +0.1 AP). A new *kind* of post-processing beats another model.
9. **The remaining pedestrian loss is box placement/size**, not missed objects (recall ceiling 0.90). Models predict a near-constant 0.75 m footprint; labels span 0.5-1.0 m. Every attempt to fix size from the points or with a learned refiner failed on real scans.
10. **The newest win: a scan-point rule.** A pedestrian box with fewer than 5 non-ground points inside gets score x0.25: +0.48 / +0.52 AP on dev150 / dev350, every half up.
11. **Do not trust local realism.** A bootstrap over 550 unlabelled dev scans said v4 beat v3 on every draw (+0.0055); the official score said v4 lost (-0.0058).
12. **Process that saved us:** adopt a change only if it improves dev150 AND dev350 AND every half, at full precision. Stop testing a variant on the second set as soon as it fails on the first.

---

## 2. Scores

### 2.1 Official submissions (Final phase, 8 allowed, 7 used, 1 left as of Nov 15 02:30)

| Version | Date | Combined | Vehicle | Pedestrian | Truck | Realism | What changed |
|---|---|---|---|---|---|---|---|
| v2 | 11-11 | 0.7997 | - | - | - | - | first final submission: levelled multi-height inference + multi-model fusion + box calibration |
| v3 | 11-11 | 0.8475 | 69.07 | 35.99 | 59.66 | 0.9423 | truck size rule + extra pedestrian TTA voters |
| v5 | 11-12 | 0.8600 | 74.30 | 35.20 | 59.58 | 0.9423 | averaged vehicle heading, truck de-dup, more voters |
| v7 | 11-12 23:33 | 0.8732 | 74.25 | 39.66 | 59.79 | 0.9423 | **pedestrian box hedging** |
| v9 | 11-13 07:30 | 0.8853 | 75.49 | 41.51 | 59.75 | 0.9507 | PV-RCNN voters from OSC, truck size hedge, **zeros trimmed to 85%** |
| v12 | 11-13/14 | 0.8875 | 75.59 | 43.68 | 59.06 | 0.9449 | + P2x3, P3x3 pedestrian PV-RCNN; per-scan levelling; **v4 realism (mistake)** |
| **v15s** | 11-14 17:57 | **0.8917** (#2) | 75.6 | 44.4 | 59.1 | 0.9507 | v13 vehicle/truck + pedestrian from v15 (P1/P3 TTA x3, P4 x1); v3 realism |

Leaderboard on Nov 14: #1 KarmaRV **0.8944** (veh 74.32, ped 41.09, truck 65.95, realism 0.9406, mAP 0.6045); #3 lukefinigan 0.8809 (realism 0.9554, higher than ours). We beat #1 on vehicle, pedestrian and realism; the whole gap is trucks (59 vs 66).
Gap to #1 after v15s: 0.0027 = about +0.9 AP in one class.

### 2.2 Candidates built but NOT submitted (all kit-valid)

| Version | What it is | Expected official |
|---|---|---|
| v10, v11 | v9 + P1 x3 + one-vote vehicle (v10); + extra voters (v11) | superseded |
| v13 | v12 predictions + v3 realism scans | about 0.8899 |
| v14s | v13 vehicle/truck + pedestrian from a build with P1/P3 TTA x3 | about 0.891 |
| **v16s** | v15s + averaged pedestrian heading | about 0.893 |
| **v17s** | v16s + pedestrian min-points rule (<5 points -> score x0.25) | about 0.8938-0.8949 (sensitivity range, not a confidence interval) |

| **v18s** | v17s + truck min-points rule (truck boxes with < 30 scan points -> score x0.25, before the truck size hedge); 8,607 of 18,512 truck boxes cut, top truck scores untouched; declaration dated 2026-11-15 | about 0.8943-0.8952 (truck part uncertain, -0.0008..+0.0009) |
| v17s_d15 | v17s with only the declaration date/tag corrected to 2026-11-15 | same as v17s |

"s" = class **splice**: vehicle and truck boxes copied byte-for-byte from v13, pedestrian boxes from the new build (see 5.6 for why).

### 2.3 Dev numbers behind the candidates (pedestrian AP, hedged; all frames / half A / half B)

| Candidate | dev150 | dev350 |
|---|---|---|
| v15s | 45.42 [45.26, 45.66] | 45.20 [46.1, 44.35] |
| v16 (avg ped heading) | 46.07 [45.86, 46.36] | 45.41 [46.29, 44.62] |
| **v17 (+ min-points)** | **46.55 [46.32, 46.87]** | **45.93 [46.85, 45.10]** |

v15s -> v17: +1.13 / +0.73 AP, i.e. +0.0032 / +0.0021 score. Applied to 0.8917 gives about 0.8949 / 0.8938.
Vehicle dev150 84.2 / dev350 84.0, all-truck 88.7 / 87.6, genuine-truck 75.0 / 73.7 are unchanged by everything since v13.

---

## 3. The scorer (read once, it explains many results)

File: `starting_kit/track_b/scoring_program/detection_eval.py`, `score.py`, `metrics.py`.
- Per frame, predictions are matched **greedily in descending score order** to the best unmatched ground-truth box with IoU >= 0.5; the PR curve is built from the **global** sort across all frames; AP is 40-point interpolated (R40: for each recall r, the max precision at recall >= r).
- **No cap on the number of predictions.** Boxes appended with scores below every existing box of their class **cannot lower AP**. Measured value of a blind "guaranteed tail": +0.07 pedestrian AP for 5x boxes, so it is useless. Hedging works differently: variants sit right under their parent and recover boxes that missed IoU 0.5 by a little.
- No KITTI difficulty filter. The leaderboard metric is **mAP over the 3 classes at IoU 0.5, R40**; the 0.7 AP is reported but not scored.
- Realism: concat the 50 synthetic scans, mask to (-40,-40,-8,40,40,2), subsample 10,000 points, compare with the 50 held-out real scans via CD, MMD, EMD, FPD (normalised). **Points at exactly (0,0,0) are kept.** The EMD term is about 92% of our remaining realism deficit.
- Official labels appear to need >= 5 sensor points: test has 19.1 pedestrians/frame; dev with the >= 5-point filter has 20.3/frame; dev without it 31.1/frame.
- Which submission counts (best or latest) is **not stated in the rules**. Check "My Submissions" on Codabench for a leaderboard toggle; otherwise assume the latest counts.
- Final-phase limits: 8 submissions total, 4 per day. The 4-8 page report is due the same day (Nov 15).

---

## 4. Foundations (before v7, short)

| Finding | Evidence |
|---|---|
| Real scans tilted about 6.5 deg (plane z = -0.006x - 0.113y - 2.87), synthetic data level | levelling per scan: mAP 15.8 -> 33.7 (SECOND mix), 37.2 (relabel) |
| Train on ground-canonical data (ground at -3.0) | `canon/build_canon.py` |
| Multi-height (6 heights -3.15..-2.775) + multi-model fusion | real-dev mAP 49.3 -> 56.3; single-seed noise about 3 points |
| Box calibration: pedestrian centre -0.045 m toward sensor then 50/50 with point centroid (+0.065 m away); vehicle height +0.09, z +0.11 | dev mAP 64.5 -> 67.7; pedestrian AP 31.9 -> 38.9 |
| Real scans hold about 15.6k points at (0,0,0) (no-return beams); range-image site twin + zeros | realism 0.87 -> 0.95 local |
| Averaged heading (vehicles) | vehicle AP 75.4 -> 80.0 dev; +5.2 test |
| Truck de-dup (buses were split into 2 fused boxes) | genuine-truck AP +4.8 dev |
| Pedestrian static down-weight (0.25 m voxels occupied in >= 30% of unlabelled dev frames): score x (1 - 0.8 x static fraction) | pedestrian AP up on dev |
| Synthetic data: CARLA/Generator V2 canonical set, site-twin set, small-truck twin, size-varied pedestrian twin, long-low truck twin | see report handoff, section 4 |

Models: SECOND, VoxelNeXt (laptop, 12-20 epochs), PV-RCNN (OSC A100, 4-6 epochs). All trained from scratch on synthetic data only.

---

## 5. What WORKED, v7 onward (in order)

### 5.1 Pedestrian box hedging (v7, +4.5 test AP)
Each pedestrian box with score >= 0.05 is also emitted as 8 size variants (0.8x0.8 and 1.25x1.25 at 0.2x score; 1.0x1.4 and 1.4x1.0 at 0.1x; 0.65, 1.6, 0.8x1.2, 1.2x0.8 at 0.05x) plus 4 diagonal 0.12 m shifts at 0.1x. Dev: 38.5 -> 42.4 (dev150), 37.1 -> 41.3 (dev350). Test: 35.2 -> 39.66.
The idea came from the leader's public result page: about 720 pedestrian boxes per frame vs our 274. Looking at rivals' public *box counts* (not their data) was legitimate and valuable.

### 5.2 PV-RCNN from OSC as voters + zero trimming (v9, 0.8853)
- PV-RCNN jobs 1 (6 ep) and 2 (seed 2, 4 ep) as pedestrian/vehicle voters: dev vehicle +1.0, pedestrian +0.4-0.9; test vehicle +1.2, pedestrian +1.9.
- **Zero trimming:** real zero share inside the scoring box is 30.8%, ours was 33.1%. Keeping 85% of the zeros: 5-seed local realism 0.9405 -> 0.9488; **official 0.9423 -> 0.9507** (+0.0084). One of the few realism changes that transferred.

### 5.3 Pedestrian-only PV-RCNN specialists (v10 -> v15s)
P1, P2, P3 (same recipe, different seeds, data `claude_canon_all_twin_st_v1`), each counted 3x as a pedestrian voter: pedestrian AP +1.1/+1.2 for P1; v12 test pedestrian 41.51 -> 43.68.
P1-TTA and P3-TTA (5 shift/yaw variants x 4 heights) x3: dev150 44.9 -> 45.3, dev350 44.5 -> 45.1 (v14s).
P4 (same plus SE-SSD-style part-dropout/sparsify/swap augmentation) x1: dev150 45.3 -> 45.4, dev350 45.1 -> 45.2 (v15s). P4 x3 failed on dev350.

### 5.4 One confidence vote per pass for vehicles (Codex, FUSE_ONE_VOTE=vehicle)
In a cluster only the best box of each (model, height) pass adds to the confidence: vehicle +0.5 / +0.4 AP dev. The same rule for trucks is much worse (-15 AP): for trucks agreement-by-count is the signal.

### 5.5 Per-scan levelling and compliance clean-up (v12)
Earlier builds levelled with the median plane of all 50 test scans, which uses other test scans. v12 fits the plane from each scan's own points (`PLANE=scan`, `codex_review/per_scan_export.py`, fixed RNG per scan, batch 1). The long-low truck voter (`second_twin_ll_*`) was removed because its data recipe referenced test-clip objects. Both came from Codex's review. Cost: small; benefit: the submission is defensible.

### 5.6 Class splice (v14s, v15s, v16s, v17s)
Adding pedestrian-only passes raises the vote denominator, which lowers truck scores and dropped about 3.6k low-score truck boxes (18,512 -> 14,939). Fix: take vehicle and truck boxes verbatim from v13, pedestrian boxes from the new build (`final/splice_ped.py`, `run_splice.sh`).

### 5.7 Averaged pedestrian heading (v16s)
Same circular-mean heading as vehicles (doubled angle, period pi), now for pedestrians too (`FUSE_YAW_CIRC_EXTRA=pedestrian`). dev150 45.42 -> 46.07, dev350 45.20 -> 45.41, every half up. Earlier (Nov 12) it was "slightly negative for pedestrians" on a weaker ensemble; with the bigger ensemble it flipped. Lesson: re-test old negatives when the ensemble changes.

### 5.8 Pedestrian min-points rule (v17s) — the newest and most useful
`calib.ped_min_points`: count non-ground points inside the box (grown 0.1 m); if < 5, score x0.25. Applied after calibrate, before hedging.
- dev150 46.07 -> 46.55 [46.32, 46.87]; dev350 45.41 -> 45.93 [46.85, 45.10]; x0.5 also passes, slightly lower.
- Why: a box with almost no scan points cannot be a scorable object (labels need >= 5 points) yet had a high score from model agreement. It is label-free at test time.
- Caveat (Codex, correct): the label rule counts points in the **ground-truth** box, ours counts points in the **predicted** box. They are different quantities; the justification is the measured gain on two dev sets, not a proof.
- It works for pedestrians but not vehicles: sparse vehicle boxes are already low-scored (see 6.5).

---

## 6. What did NOT work (by area)

### 6.1 More pedestrian models
| Tried | Result |
|---|---|
| P5 x3 (plain recipe, different data) | dev150 45.3 [44.9, 45.8] fail; dev350 45.2 |
| P6 x1 / x3 (aug recipe, different data) | dev150 45.4/45.4; dev350 45.0/44.9 fail |
| P4 x3 | dev350 fail |
| P4 TTA x1 / x3 | dev150 45.4 / 45.5; dev350 45.1 / 44.9 fail |
| P2 TTA x3 | 45.1 fail |
| **Voxel R-CNN VR1 / VR2 (new architecture), x1 / x3 / both** | dev150 base 46.1 -> 46.0-46.2; VR2 alone 42.4 (a strong single model) but adds <= +0.1; VR1 x3 lower on half A. dev350 not finished (stopped, nothing to rescue) |
| Part-A2 pedestrian | crashed at the first step (CUDA device-side assert in the point-part loss); not diagnosed |
Why: the ensemble already finds every pedestrian these models find. The loss is box quality and ranking, not recall (ceiling 0.90).
Cost: VR1 about 1.6 A100-hours, VR2 about 1.0 A100-hour (est.), local Voxel R-CNN L1 trained 2 of 5 epochs (about 49 min/epoch on the RTX 4070) before being stopped.

### 6.2 Pedestrian fusion settings (all on the v15/v16 base)
| Variant | dev150 | dev350 |
|---|---|---|
| one vote per pass for pedestrians | same | same |
| cluster radius 0.3 / 0.5 (default 0.4) | 45.3 / 45.4 | 45.1 / 45.1 |
| static down-weight 0.6 / 1.0 (default 0.8) | 45.3 / 45.4 | 45.1 / 45.2 |
| weights score^2 | -0.07 | not run |
| weights score^0.5 | +0.08 | +0.02, one half -0.01 (noise) |
| geometry from top-3 / top-6 members only | 43.5 / 43.9 | not run |
| equalising swapped length/width-heading forms before averaging (Codex) | 45.7 | 45.3 |
| rank by agreement with per-checkpoint boxes, lambda 0.25 / 0.5 (Codex) | 45.98 / 45.98 | not run |
| robust z/height, median bottom/top: replace / 50-50 blend (Codex) | 45.84 / **46.25** | 50-50: +0.02, half B -0.02 -> fail |
| no sideways centroid shift in refinement / half (Codex) | 45.77 / 45.89 | not run |
Audit of the alignment idea (dev150): 30% of cluster members (456k of 1.51M) and 44% of clusters contained a member in the 90 deg-swapped description. The problem was real, but fixing it lost AP: much of that "disagreement" is genuine, and our circular mean (period pi) already tolerates the 180 deg flip.
Top-3/top-6 failed because averaging many boxes *is* the benefit: ensemble averaging reduces centre noise.

### 6.3 Hedge variants (new and re-tuned)
- Re-tune of the 13 variants on v15 and v16 (factors x0.5/x1.5/x2, smin 0.02/0.1, mid sizes 0.9/1.1, diagonal shifts 0.2, radial shifts +-0.1/0.15/0.25 toward/away from the sensor): the **current hedge is best**; smin 0.02 ties (46.07 vs 46.07).
- Extra copies on the v16 base: rotated 90 deg (f.1 / f.2), +-45 deg, height x0.85/x1.15, z +-0.1: all -0.1 to -0.8.
- Point-guided hedging on v17: apply the 5-point rule to each variant's own count (46.54, no change); graded rule 5-7 points x0.7 (46.28, worse); scan-points choosing among variants and promoting the 0.8x box (see 10).

### 6.4 Rules from point features (on the v16 base, dev150)
| Rule | AP (base 46.07) |
|---|---|
| pts<5 x0.5 / x0.25 | 46.50 / **46.55** (kept) |
| pts<10 x0.5 / x0.25 | 45.30 / 41.06 |
| top point height < 0.8 / 1.0 m | 46.07 / 46.06-46.02 |
| vertical extent < 0.5 / 0.8 m | 46.11 (x0.5) / 45.63 |
| spill (share of points just outside the box) > 0.3 x0.7 / x0.5 | 46.08 / 45.67 (on v17: 46.08 vs 46.55) |
| spill > 0.5 x0.7 / x0.5 | 46.54 / 46.50 (no change) |
| point-centroid off-centre > 0.5 | 46.56 (+0.01, noise) |
Typical in-box point counts (top 5000 boxes, dev150): TP median 18, localisation-FP 13, background-FP 6. So point count separates background from objects but not well-placed from misplaced boxes.

### 6.5 Vehicle min-points (dev150 / dev350, base 84.22 / 83.96)
pts<5 x0.5: 84.30 / 83.97; x0.25: 84.21 / 83.91 (nothing). pts<10: 83.6 / 83.4 (x0.5), 82.5 / 82.4 (x0.25). pts<20: 83.4 / 83.1, 81.2 / 81.1. Sparse vehicle boxes already score low; down-weighting more only drops true far cars.

### 6.6 Trucks (the unsolved class; the user said leave it)
- Dev truck AP is inflated by about 13 parked objects in every frame (garbage truck "TrashCan" 8 m, buses): 202 of 258 dev150 truck labels are parked repeats; we score about 93 AP on them. Non-parked trucks 64.1/63.1 on dev, close to test 59.1. Box truck + van AP 40/30: they are detected (IoU >= 0.5 for 34/34 and 78/81) but score only about 0.07.
- Re-tuning the truck rule on realistic metrics: nothing beats a 0.5 / base 0.1. The "base copies drown box trucks" hypothesis was wrong (lowering base hurt).
- Failed truck ideas (neutral or worse on test even when dev looked good): long-low synthetic set + SECOND, small-truck SECOND voters x2/x3, PV-RCNN truck voter, truck-only specialists T1/T2, size hedging (40% fraction: -9 to -10 AP), score boost for long boxes (+1.0 dev150, -0.1 dev350), cutting low-score clutter, lower vehicle-to-truck copy scores, VoxelNeXt as truck voter (-13 to -15), alternative fused truck scores (max/mean/geometric -9 to -27), one-vote truck scoring (-15 to -17), tighter box fitting (-5.9/-4.6), learned truck fix (-23), per-class score normalisation (-1.15 genuine).
- Label-free test-clip diagnosis: about five fixed large objects (an 8 m trash-can-like object at about 26 m always found; two 8-9 m vehicles at 25-29 m found but scored only 0.22-0.31 behind false boxes). We deliberately wrote no location-specific rule.
- Excluded permanently: the long-low voter (rule), and pvrcnn_truck_T1 (trained on long-low data).

### 6.7 Realism
- The local proxy is **biased**: bootstrap over random 50-frame references from 550 unlabelled dev scans (dev150 + dev350 + cw1) with random subsample seeds: v4 - v3 = +0.0052 to +0.0058, sd 0.0007, P(v4>v3) = 1.00 over every draw; official: v4 0.9449 < v3 0.9507. The official reference differs from every dev pool in some way we cannot see; matching it would mean tuning on test scans (not allowed).
- The test scans are raw scans cropped to the range box and capped at 50,000 points with zeros kept; point format is not the cause.
- Things that did nothing locally and were not worth testing officially: scan swapping (greedy 400-candidate search, selection-half gain did not hold on the confirmation half), near-range thinning, blending per-scan zero counts toward real quantiles (worse on all 6 checks), range x height thinning.
- Decision: keep v3 scans. No more realism experiments.

### 6.8 Earlier pedestrian-size attempts (do not retry)
Size-varied synthetic pedestrians and size-aware fine-tunes (predicted length stayed 0.74-0.84), point-extent size rules (19 settings, neutral to -22), rider-size boxes (worse in 5 settings), learned PointNet refiner (synthetic centre error 0.10 -> 0.03 m but real 38.5 -> 35.3), pedestrian height/z hedging (all lower), range-based score boosts, more vehicle seeds (saturated).

---

### 6.9 Round 3-4 results (Nov 15, after v17s; all pedestrian AP, hedged, full precision, v17 base 46.554 [46.319, 46.867] / 45.929 [46.851, 45.098])
| Idea | dev150 | dev350 | Verdict |
|---|---|---|---|
| Codex: suppress duplicate parents sharing >= 80% of their scan points (x0.5 / x0.25); 2182 / 5080 parents flagged | 46.555 / 46.552 | 45.921 / 45.921 | nothing |
| Point-guided hedge scores A1 (alpha 2) / A2 (alpha 4) | 46.53 / 46.40 | 45.98 / 45.88 | A1 +0.05 on dev350 but -0.02 on dev150 -> fail |
| **Promote the 0.8x variant when it keeps >= 95% / 90% of the points (S1 / S2)** | **39.0 / 35.6** | **38.1 / 35.2** | catastrophic: the 0.8x box almost always keeps the points (they sit at the centre), so most primaries were swapped |
| Density-normalised point rule n*(r/20)^2 < 8 x0.5 / < 16 x0.5 / < 8 x0.25 | 46.53 / 46.00 / 45.89 | 45.79 / 45.47 / 45.09 | worse |
| **Primary box size x0.95 / 0.90 / 0.85** | 46.26 / 43.49 / 38.79 | 45.80 / 42.55 / 37.96 | worse |
| **Primary box size x1.03 / 1.05 / 1.10** | 45.59 / 44.80 / 41.89 | 45.13 / not finished / not finished | worse |
| Point-count definition: non-ground grow 0 (C1) / all points grow 0 (C2) / all points grow 0.1 (C3) | 46.567 / 46.593 / 46.545 | 45.927 / 46.035 / 45.989 | C2 +0.04 / +0.11, but dev150 half B -0.06: noise-level, not adopted |
| Score vs range at fixed score (diagnostic, dev150 top 6000): TP rate near (0-15 m) / mid / far | middle score tercile 0.40 / 0.26 / 0.34; high tercile 0.64 / 0.70 / 0.56 | - | not monotone -> a one-constant range factor cannot help; rules not run |
**Density check (so we do not rebuild the training data for nothing).** `coop/density_gap.py`: points inside the box, synthetic training objects vs real dev GT, by range. Pedestrian median points (synthetic / real): 0-10 m 83/60, 10-15 m 47/62, 15-20 m 27/30, 20-30 m 13/18, 30-45 m 7/7. Vehicle: 689/1141 (0-10 m), 401/318, 204/195, 83/76, 30/26. Truck 15-20 m 850/813, 20-30 m 302/319. So no obvious gross point-count mismatch was found (synthetic pedestrians are slightly sparser at 10-30 m, ratio 0.72-0.90); rebuilding the data is not justified before the deadline. This does NOT prove the data is adequate: visibility, shape, occlusion and annotation geometry can still differ, and the synthetic counts were not filtered to >= 5 points like the real ones (Codex's correction).
**v17 decomposition (Codex idea 1, dev150):** score reduction alone (A) = 46.554, same as v17; hedge-family removal alone (B) = 45.977 (below even no-rule 46.065). So the whole gain comes from lowering the primary and variant scores, not from removing hedge families. **Static double penalty (Codex idea 2, dev150):** softening the static penalty for point-flagged boxes (D2 0.4) = 46.553, no change -> closed. Also: the released test scans have the same density as dev inside the range box (median 34,458 vs 34,869 non-zero points; 29,479 vs 29,825 within 25 m), so the 5-point rule is not biased by thinner test scans.

Takeaways:
- **The primary box size is a sharp optimum at x1.00** (-0.3 for 5% smaller, -1.0 for 3% bigger, -3 for 10% either way). The hedge family was designed around 0.75 x 0.75, so scaling the primary moves the whole family off its sweet spot. The "labels are 0.65 m, ours 0.75 m" gap cannot be closed by a global scale; per-box choice among candidates is what hedging already does.
- Promoting a smaller candidate is dangerous: it is not a free re-ranking, it changes which box carries the top score for almost every pedestrian.
- Any rule that sits on top of v17s now adds < 0.1 AP. The cheap post-processing seam in the pedestrian pipeline is exhausted; what remains is placement/size inside the models, which every direct attempt has failed to improve.

**Truck min-points rule (REPORT ONLY).** Truck boxes after the truck rule + dedupe: 41% have < 10 points inside, 72% < 30, 89% < 100 (most are the low-score vehicle copies). `pts<30 x0.25`:
| Truck metric | dev150 | dev350 |
|---|---|---|
| non-parked (REAL) | 64.13 -> 64.63 [halves +0.23, +0.47] | 63.08 -> 63.27 [+0.18, +0.02] |
| Truck + Van | 39.81 -> 41.27 [+1.5, +1.3] | 30.34 -> 30.81 [+0.4 .. -0.2] (x0.5: 30.89, halves +0.9, +0.6) |
| all-truck (old metric, parked-inflated) | 88.72 -> 88.97 | 87.56 -> 87.62 (half B -0.26 for x0.25) |
`pts<10` is about +0.05; `pts<100` is clearly harmful (-4 to -6 on non-parked). Value if it transfers: +0.2 to +0.5 AP = +0.0006 to +0.0015; worst observed -0.26 AP. Not in any zip; the user decides.

---

## 7. Anatomy of the remaining pedestrian loss (dev150, v16 base)

`coop/ped_fp_diag.py`, `coop/ped_loc_diag.py`. Recall ceiling 0.903, 106,528 predictions (primaries + hedge).
AP if every false positive of one type were removed (upper bounds, not achievable):

| FP type | AP after removal |
|---|---|
| none removed | 46.07 |
| duplicate of an already matched GT | 47.26 |
| near a vehicle/truck | 46.21 |
| on a < 5-point pedestrian GT | 47.05 |
| background | 51.63 (v17 recovers about 0.5 of the 5.6) |
| **localisation (IoU 0.1-0.5 to a scorable GT)** | **60.09** |

Among the top 3000 primaries there are 1702 TPs and 729 localisation FPs. Copying one component from the GT fixes this share of those 729: length/width 50%, centre xy 48%, z+height 35%, z 13%, height 9%, yaw 7%. Median errors: centre about 0, bottom about 0, top +0.04 m, length +0.11 (but mean -0.02, so bimodal), width +0.07. GT median size 0.65 x 0.67 x 1.73 vs predicted 0.76 x 0.74 x 1.77.
Interpretation: placement and footprint of already-found pedestrians is where the points are, but every direct size/centre fix (points, learned refiner, retraining) failed on real scans. The only things that moved AP were ranking/hedging tricks and averaging.

---

## 8. Mistakes and time wasted (so we do not repeat them)

| What | Cost | Lesson |
|---|---|---|
| **Adopted v4 realism scans because the local proxy said +0.005** (v12) | official realism -0.0058, cancelled about half of the pedestrian gain; wasted a submission slot | a local metric that cannot be validated is a coin flip; keep what was officially confirmed |
| Predicted v12 would score about 0.892; got 0.8875 | credibility | state ranges and name the unvalidated part |
| Trusted dev truck AP (88) | many hours on truck ideas that did not move test (59) | gate each class on what has transferred before; use non-parked / box-truck metrics |
| Hypothesis "truck base copies drown box trucks" (lowering base) | an hour | test the cheapest hypothesis first; it was wrong |
| v14 truck dilution (18,512 -> 14,939 trucks) | found by diffing counts | always diff per-class box counts against the previous zip; fixed with the class splice |
| P4/P5/P6, Voxel R-CNN VR1/VR2/L1, Part-A2 | about 6 A100-hours + several laptop GPU hours, ~0 AP | a new model type was the right bet; but at +0.1 on dev150 stop and do not spend more |
| Realism work (bootstrap, thinning, swapping) | several hours | after the v4 inversion nothing local can be validated |
| OOM (WSL 11 GB) when two evaluations ran in parallel with a gate | restart | run fewer jobs in parallel; sequential evaluations |
| `pkill -f` / `TaskStop` killed neighbouring jobs (P38 restarted, 6 min lost; earlier waiters matched their own command line) | minutes | kill by exact PID or script name; never pattern-match your own command |
| PowerShell inline Python/ssh quoting failures; WSL launch timeouts | minutes each | every command lives in a script file; run with `wsl -e bash -lc "bash script.sh"` after `sed -i 's/\r$//'` |
| Wrong environment for a script (no POT module for realism) | one failed run | the realism scripts need `PYTHONPATH=$HOME/projects/roadside-pilot/realism_deps` and the bdc2026-track-b env |
| A second AI (Codex) given the wrong config values earlier queued a wasted 12-epoch run | cancelled | read the actual config before describing it |
| Stale logs: P35's first runs printed rounded one-decimal AP, hiding +0.1 effects | ambiguous gates | report two decimals (`coop/ped_ap2.py`) |
| Asking for "everything on OSC" for CPU re-scoring tests | I advised against (setup 1-1.5 h vs 15 min saved) | OSC is for GPU training; CPU replay of cached predictions is faster on the laptop |

---

## 9. General lessons

1. **Score arithmetic first.** Rank every idea by expected combined gain: +1 pedestrian AP = +0.0029, +0.01 realism = +0.004.
2. **Transfer table:** pedestrian dev -> test is about 1:1; vehicle partial; truck none; realism none (inverted). Spend effort on the class whose dev predicts test.
3. **Gate:** adopt a change only if both dev sets and all four halves improve, at full precision. Repeated tuning on the same scenes still has selection risk, so the halves are not independent replications.
4. **Cheap before expensive:** test post-processing from cached predictions (minutes) before any training (hours). Training added about nothing after P3.
5. **Stop early:** if a variant fails on dev150, do not run it on dev350.
6. **Reproduce the baseline** after any speed-up. The fast pedestrian-only path (vehicle/truck passes kept as empty "none" passes so the vote denominator is unchanged) reproduced the full run exactly (46.07 / 45.86 / 46.36) before we trusted it.
7. **Check box counts per class** between zips (`diff` of class totals), not only AP.
8. **Averaging is strong.** Fusion by agreement, circular-mean heading, multi-height, TTA: all ensemble-style fixes won; every attempt to *replace* the average with a single "better" box (top-k, learned refiner, point-fit sizes) lost.
9. **A rule derived from the label definition can generalise better than a tuned constant** (the 5-point rule), but the caveat stands: it was accepted because it passed the dev gate, not because of the definition.
10. **Retest old negatives when the system changes** (averaged heading was negative for pedestrians on the small ensemble, positive on the big one).
11. **Read rivals' public pages** for box counts and per-class AP: the hedging idea came from there.
12. **Get an independent review** (Codex): found one-vote vehicle scoring, the per-scan compliance issue, the long-low rule, the aliasing idea; its code-reading caught real flaws (rounded APs, rewrite-count != disagreement). It also proposed ideas that failed; both outcomes are information.
13. **Submit with margin.** Codabench had outages (Oct 21-26, Nov 6); do not submit at 23:00 on the last night.

---

## 10. Open / in progress at the time of writing (Nov 15 02:25)

| Idea | Status |
|---|---|
| All of the round 3-4 pedestrian ideas (see 6.9) | **closed, none passed the gate**; v17s is still the best pedestrian candidate |
| Truck min-points rule `pts<30 x0.25` (REPORT ONLY; the user said leave trucks) | small positive on realistic truck metrics, one old-metric half -0.26; needs the user's explicit OK before it goes into any zip (see 6.9) |
| Codex's promote-smaller-box with >= 10-point parent and 100% retention | to run only if S1/S2 show promise |
| Stack every passing rule, re-check on both dev sets and all halves, build v18s | pending |
| Report (4-8 pages) | due Nov 15; handoff text in `REPORT_HANDOFF_FOR_CHATGPT.md`; must disclose the real-data use list (unlabelled dev clouds for the static map, the site twin and calibration constants chosen on dev labels) |

Decision rule for the last submission: submit the best candidate whose dev gain passes the gate on both sets. If nothing beyond v17s passes, v17s is the final; v16s is the fallback; v15s (0.8917) is the known score.
The declaration inside each zip says `Date: 2026-11-14`; correct it to 2026-11-15 in the final zip.

---

## 11. Where things are (quick map)

| Need | File |
|---|---|
| Fusion + env switches | `final/fuse_test.py` |
| Calibration, hedging, min-points, truck rule, dedupe | `final/calib.py` |
| Build a candidate | `final/build_candidate_v7.sh` (env `PED_MINPTS=1`, `FUSE_YAW_CIRC_EXTRA=pedestrian`), `final/run_v14.sh`, `final/run_v16.sh`, `final/run_v17.sh` |
| Class splice | `final/splice_ped.py`, `final/run_splice.sh <ped-src> <out>` |
| Kit validation | `final/kit_validate.sh`, `final/verify_zip*.py` |
| Fast pedestrian eval (full precision) | `coop/eval150.py`, `coop/ped_ap2.py`, `coop/verify_hedge_fn.py`, `coop/verify_minpts.py` |
| Diagnostics | `coop/ped_fp_diag.py`, `coop/ped_loc_diag.py`, `coop/truck_real.py`, `twin/realism_boot.py` |
| Gates and logs (tonight) | `final/p32..p41*.sh/log`, `coop/ped_*.log`, `coop/minpts_more*.log` |
| Last zips | `C:\Users\arnav\Downloads\bdc2026_trackB_submission_v15s.zip` (submitted), `v16s.zip`, `v17s.zip` |
| OSC scripts (never run without the user's per-job yes) | `osc/osc_submit_*_approved.sh`, `osc/osc_VR_status.sh` |
| Handoffs | `HANDOFF_2026-11-12_FINAL_PHASE.md`, `REPORT_HANDOFF_FOR_CHATGPT.md`, `CODEX_BRAINSTORM_PROMPT.md` |

Rules that never change: synthetic-only training; never the 100 forbidden frames; each test scan processed independently; nothing tuned on test scans; no location-specific rules; no label probing; the long-low voter stays out; Claude never submits to Codabench and never runs sbatch/scancel on OSC without an explicit per-job yes.
