# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 92a605a, "Full-specimen FIN-Benthic2 and AquaMonitor geometry on LUMI", with the author's
decisions D1–D3 on the Codex Astra methods review (author, 2026-11-11: all three recommended options).
**Status: corrections applied and the CPU analysis rerun on LUMI for all five folds of both datasets, reusing the
existing embeddings (no GPU). Stopped for the author's review.** No test evaluation, BIOSCAN, DF24 or classifiers.

Code: corrections e0f39a5 (run on LUMI), per-class summary columns in the commit of this response. Evidence (corrected):
`docs/evidence/{finbenthic2,aquamonitor}_full_geometry/{folds,summary,descriptive_fold0,baseline_capped_fold0}` and
`docs/evidence/full_geometry_comparison.md`. The first run (before the review) is kept unchanged in
`docs/evidence/*_full_geometry/first_run_before_review/` and `docs/evidence/full_geometry_comparison_first_run_before_review.md`.

## 1. What changed (D1–D3)

| Decision | Change | Where |
|---|---|---|
| D1 | Dataset-level averages over **reliable fits only** (the mixture and one Gaussian both `ok`: eligible, converged, every component's responsibility mass ≥ d+1), with taxa included / excluded per comparison. All-fit averages kept as labelled diagnostics. All models also compared on the **same taxa** (classes where every model is reliable). | `src/statistics/class_macro.py`, `class_geometry_pilot.py` `summarise` |
| D2 | Per-taxon label from **one model**, the reliable 2-component tied mixture: "mixture-supported split (exploratory)" only if its own held-out interval lies above 0 **and** its own split is stable. Other labels: "single Gaussian predicts better", "inconclusive", "insufficient evidence". Other mixtures reported separately (`other_mixtures_ci_above_0`). No claim of biological modes. | `class_modality.split_verdict` |
| D3 | Stability: a failed or under-supported resample (a component below d+1 on its subsample) counts as ARI 0, median over all 200 resamples. Source-model eligibility with its own parameter count. n = 30 summaries only for taxa with ≥ 30 specimens. Reliability flag in output A. Metadata per specimen camera-balanced like the embeddings; wording "partition agreement" / "rank separation"; train and val association counts both reported. Five-fold summary carries taxa counts, "not analysed" entries, per-fold support and pair coverage. Test rows: "dropped before the feature lookup, never used" (their features are read with the cache and discarded). | `class_modality.py`, `specimen_geometry.py`, `descriptive_geometry.py`, `component_metadata.py`, `summarise_five_folds.py` |

Regression tests (full suite 67 passed): reliable-only vs all-fit summary, common-taxa comparison, same-model label,
source-model budget, exact n = 30, five-fold per-class columns. The stability test separates old from new behaviour:
for a "split" that only isolates two outlier specimens the old code gave median ARI 1.0, the corrected code 0.

## 2. LUMI rerun (project 462001534, code e0f39a5, small partition)

| Job | What | Elapsed | Result |
|---|---|---|---|
| 23652423 | FIN-Benthic2 five folds (8 cores each) | 4:36–6:10 | completed |
| 23652424 | AquaMonitor five folds (16 cores each) | 8:03–9:05 | completed |
| 23652425 | output A, montages, capped fold-0 baselines rerun with the corrected code, LUMI-vs-Dell, five-fold summary | 12:07 | completed |

35.7 CPU-hours (sacct, logical CPUs). Earlier steps of this task: extraction 1.9 GPU-h (jobs 23642052, 23642060) and
the first CPU run (114.4 CPU-hours incl. 60 in a hung attempt). The per-class columns of the five-fold summary were
added after the run and recomputed on the Dell from the fetched outputs (read-only aggregation, seconds); every other
summary file is byte-identical to the LUMI one.

## 3. Dataset-level results (held-out log density per val specimen, class-macro; reliable fits only)

**2-component tied mixture minus one Gaussian**, per fold: mean [95 % interval], taxa included / fitted:

| | FIN-Benthic2, 8 PCs | AquaMonitor, 8 PCs |
|---|---|---|
| fold 0 | +0.117 [+0.050, +0.177], 32/36 | +0.270 [+0.198, +0.336], 39/41 |
| fold 1 | +0.060 [−0.013, +0.126], 33/35 | +0.321 [+0.269, +0.374], 39/41 |
| fold 2 | +0.082 [+0.024, +0.136], 34/34 | +0.262 [+0.195, +0.323], 39/40 |
| fold 3 | +0.000 [−0.093, +0.081], 34/35 | +0.275 [+0.212, +0.334], 39/41 |
| fold 4 | +0.028 [−0.065, +0.123], 33/35 | +0.261 [+0.191, +0.328], 38/41 |
| fold mean [min, max] | +0.057 [0.000, 0.117]; interval above 0 in 2 folds | +0.278 [0.261, 0.321]; above 0 in 5 of 5 |
| all fits (diagnostic) | −0.057 [−0.169, 0.082] | +0.266 [0.206, 0.328] |
| capped fold-0 pilot, corrected | +0.119 [+0.049, +0.184] | +0.621 [+0.550, +0.683] |

At 16 PCs: FIN-Benthic2 +0.011 [−0.122, 0.071], no fold interval above or below 0; AquaMonitor +0.516 [0.399, 0.629],
above 0 in 5 of 5 (taxa included 30.2 and 32.2 of the fitted taxa on average; more fits are unreliable at 16 PCs).

**Source-dataset model (AquaMonitor only)** minus one Gaussian, 8 PCs: +0.494, +0.426, +0.403, +0.490, +0.462 per fold
(all 41 or 40 taxa; intervals above 0 in 5 of 5); 16 PCs fold mean +0.963.

**All models on the same taxa** (taxa where every model is reliable; 8 PCs, fold mean [min, max]):

| | FIN-Benthic2 (23.4 taxa) | AquaMonitor (30.2 taxa) |
|---|---|---|
| 2-component tied | +0.193 [0.155, 0.219] | +0.327 [0.263, 0.366] |
| 3-component tied | +0.266 [0.239, 0.316] | +0.571 [0.415, 0.656] |
| 2-component full | +0.238 [0.187, 0.293] | +0.337 [0.202, 0.471] |
| 3-component full | +0.061 [−0.043, 0.147] | +0.243 [0.078, 0.331] |
| source-dataset model | — | +0.517 [0.415, 0.581] |

**Reading.** With reliable fits only, FIN-Benthic2 shows a small mixture gain (fold mean +0.057 at 8 PCs; none at
16 PCs), not the slightly negative value of the first run; it remains about a fifth of AquaMonitor's (+0.278), and on
the same taxa about 0.6 of it. AquaMonitor's gain holds in every fold but is less than half the capped pilot's; the
source-dataset model predicts better than any unsupervised mixture in AquaMonitor. On the common taxa the FIN-Benthic2
gain is larger than over all reliable taxa, because the common set leaves out taxa whose larger mixtures are
unreliable; neither set is a random sample of taxa.

## 4. Per-taxon results (labels and 2-component tied difference over folds)

Full table, one row per taxon and PCA dimension, with each fold's label, difference, interval and train / val
specimens: `docs/evidence/*_full_geometry/summary/per_class_labels.csv`.

| Labels at 8 PCs (fold mean [min, max]) | FIN-Benthic2 (34–36 taxa) | AquaMonitor (40–41 taxa) |
|---|---|---|
| mixture-supported split (exploratory) | 7.6 [6, 9] | 21.0 [18, 24] |
| inconclusive | 23.0 [20, 25] | 16.8 [15, 21] |
| single Gaussian predicts better | 2.6 [2, 3] | 1.0 [0, 2] |
| insufficient evidence | 1.8 [0, 4] | 2.0 [1, 3] |
| supported in all 5 folds | *Elmis aenea*, *Micrasema gelidum*, *Oulimnius tuberculatus* | 11: *Asellus aquaticus*, *Caenis horaria*, Ceratopogonidae, Chironomidae, *Kageronia fuscogrisea*, *Leptophlebia*, *Micronecta*, Oligochaeta, *Oulimnius tuberculatus*, *Psychomyia pusilla*, Tanypodinae |

Largest 2-component tied gains, 8 PCs: fold mean [min, max], folds with interval above 0, folds supported:

| FIN-Benthic2 | | AquaMonitor | |
|---|---|---|---|
| *Limnius volckmari* | +1.200 [0.261, 1.624]; 5; 3 | *Micronecta* | +0.917 [0.853, 0.974]; 5; 5 |
| *Oulimnius tuberculatus* | +1.091 [1.007, 1.179]; 5; 5 | *Centroptilum luteolum* | +0.762 [0.487, 1.095]; 4; 4 |
| *Elmis aenea* | +1.060 [0.829, 1.137]; 5; 5 | *Caenis rivulorum* (reliable in 3 folds) | +0.718 [0.341, 1.040]; 2; 2 |
| *Oxyethira* sp. | +0.526 [0.391, 0.674]; 4; 4 | *Psychomyia pusilla* | +0.663 [0.607, 0.722]; 5; 5 |
| *Micrasema gelidum* | +0.475 [0.397, 0.560]; 5; 5 | *Stylaria lacustris* | +0.622 [0.461, 0.757]; 4; 4 |
| *Micrasema setiferum* | +0.321 [0.156, 0.459]; 3; 3 | *Leptophlebia* | +0.479 [0.436, 0.515]; 5; 5 |

- The three riffle-beetle taxa (larva vs adult in the fold-0 montages) remain the strongest FIN-Benthic2 groups;
  *Limnius volckmari* beats one Gaussian in all 5 folds but its split is stable in only 3.
- Taxa whose 2-component interval is below 0 in at least one fold (8 PCs): 10 FIN-Benthic2 taxa (e.g. *Agapetus* sp.,
  *Polycentropus irroratus*, *Rhyacophila nubila*), 4 AquaMonitor taxa (Limnephilidae, *Limnephilus*, *Sialis*,
  *Sisyra*).
- Supported taxa that also follow metadata (train, 8 PCs, fold mean): AquaMonitor 11.6 of 21.0 have source-dataset ARI
  ≥ 0.5; FIN-Benthic2 3.4 of 7.6 have crop-size AUC outside 0.2–0.8. Val counts are the same means (11.6, 3.4).

## 5. Unchanged by the corrections

- Between taxa (fold means): pairs with val d′ < 2: 1.6 % FIN-Benthic2, 1.8 % AquaMonitor; closest pairs in every fold
  as before (*Polycentropus flavomaculatus* / *P. irroratus*, *Baetis*, *Elmis* / *Oulimnius*; *Cyrnus trimaculatus* /
  *Polycentropus flavomaculatus*, *Caenis horaria* / *C. luctuosa*); Spearman of pair d′ between folds, median 0.91
  and 0.79.
- Both camera views in the same component (median, 8 PCs): 0.828 vs 0.967.
- Feature checks: CPU vs GPU ≤ 2.1e-5; LUMI vs Dell features (AquaMonitor, 196,983 images) max 4.8e-5.

## 6. Output A (exploratory, in-sample: fold-0 train + val pooled, fold-0 test not used)

FIN-Benthic2 7,707 specimens / 38 taxa; AquaMonitor 33,732 / 42. Reliable 2-component fits at 8 PCs: 34 of 38 and
40 of 42; in-sample BIC prefers the 2-component tied model in 22 of 34 and 38 of 40 of them; both views in the same
component 0.83 vs 0.97. Not held-out evidence.

## 7. Limits

- Five folds share specimens: fold ranges show consistency across assignments, not replications. Averaged interval
  endpoints are descriptive. Intervals are pointwise and conditional on the fitted PCA, models and included taxa.
- Reliable-only averages use different taxa per model; the common-taxa table fixes the taxa but is a selected subset.
- A better-predicting mixture with a stable split can still reflect skew, heavy tails, acquisition or pose; ARI and AUC
  show partition agreement and rank separation, not cause. Life stage is not annotated; crop area is not size.
- Numerical tie: in the AquaMonitor capped baseline, *Mystacides*' source-model interval has its lower end at 0 within
  rounding (+1.3e-15 on the Dell, −6.3e-15 on LUMI), so counts of intervals above 0 can differ by one between machines.

## 8. Methods review by Codex Astra (high effort, read-only)

**Asked** (at the author's request, after the code was written): review of the mathematical and statistical methods
only — specimen unit; global whitened PCA and held-out density comparison; bootstrap and class-macro interval; verdict
and split stability; source model; d′, radius separation, subspace overlap, effective rank with unequal class sizes;
five-fold aggregation; output A; metadata measures. Handed over: `task.md`, the previous response, the analysis code
and a context note (`Feature_geometry_data/full_specimen/astra_methods_review/context.md`). Record
`~/agent_consults/2026-11-11_183354_codex_gpt-6-astra_high.md` (session `01a1cb75-723b-7eb0-ba3a-10db70d181be`).

**Found:** "needs changes — the core density comparison is sound". Six findings, all confirmed in the code:
(1) averages included fits marked unreliable; (2) the label mixed models and its stability median ignored failed
resamples; (3) n = 30 summaries used fewer specimens for small taxa; (4) source-model eligibility used the wrong
parameter count; (5) the five-fold summary dropped denominators; (6) metadata wording overstated ARI / AUC. Plus my
"never loaded" wording. Impact on the first run: `docs/evidence/full_geometry_reliability_audit.json`.

**Done:** all proposed changes (section 1), decided by the author as D1–D3; results above are the corrected ones.

## 9. Open items for the author

- **Code cleanup (asked in chat, not yet done):** two unused imports in older files; three overlapping cache-comparison
  scripts that could become one; `class_geometry_pilot.py` (390 lines, fitting + summaries + plots + montage
  selection) to be split into modules shared with `descriptive_geometry.py` and renamed; `audit_mixture_reliability.py`
  only reads pre-correction outputs; `src/statistics/gaussian.py` and `src/lightning_utils/` are used only by their
  tests (kept on the author's earlier instructions). Proposed as a separate task with a check that outputs are
  reproduced exactly.
- Next research steps (folds of other datasets, life-stage labels, BIOSCAN-5M / DF24) need a new task.

## 10. Rerun

```bash
# LUMI (code dir C; embeddings already in /scratch/project_462001534/feature_geometry/full_specimen/<dataset>/extract)
sbatch --array=0-4 --cpus-per-task=8 --mem=32G --time=00:45:00 scripts/lumi/geometry_folds.sh C finbenthic2 OUT
sbatch --array=0-4 scripts/lumi/geometry_folds.sh C aquamonitor OUT
sbatch --dependency=afterok:<both arrays> scripts/lumi/geometry_summary.sh C OUT
# selection and extraction: see commit d072185 (scripts/build_full_specimen_selection.py, scripts/lumi/extract_dinov3_l.sh)
```
