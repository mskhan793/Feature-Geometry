# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 92a605a, "Full-specimen FIN-Benthic2 and AquaMonitor geometry on LUMI".
**Status: all planned runs done (extraction, five folds × two datasets, output A, summary). Stopped for the author's
review, with three decisions open (section 6) from a methods review by Codex Astra (section 5).** The decisions change
how the dataset-level averages and the per-taxon label are computed, so the headline numbers in section 3 are given
both as computed and under the reviewer's correction. No test evaluation, BIOSCAN, DF24 or classifiers.

Previous response (FIN-Benthic2 fold 0, task 366ee62): commit 6c4c515, `docs/evidence/finbenthic2_geometry/`.

## 1. Preflight and selection (unchanged since commit d072185)

One global, fold-independent selection per dataset: every eligible specimen, up to 10 frames per specimen and camera,
spread over imaging runs and capture time (the pilots' rule; earlier fold-0 cohorts are exact subsets). Evidence
`docs/evidence/*_full_geometry/selection/`.

| | FIN-Benthic2 (derived five-split CSV `1c18f058…`) | AquaMonitor Classification (official folds, manifest `d6e63bb6…`) |
|---|---|---|
| Eligible images / specimens / classes | 460,004 / 9,631 / 39 | 2,708,023 / 42,165 / 42 |
| Specimens per fold train / val / test | 6,160–6,165 / 1,540–1,543 / 1,924–1,928 | 29,346 / 4,386 / 8,433 |
| Selected images | 186,284 | 836,159 |
| Train specimens per class (min / median / max) | 8 / 143 / 567 | 31 / 129 / 7,436 |
| New images encoded | 57,347 (128,937 reused from fold-0 job 23623689) | 836,159 |

## 2. What ran on LUMI (project 462001534)

| Job | What | Resources | Elapsed | Checks |
|---|---|---|---|---|
| 23642052 | FIN-Benthic2 extraction (code d072185) | dev-g, 1 node, 8 GCDs | 7:34 | 57,347 / 57,347 rows; CPU vs GPU max abs diff 1.8e-5 |
| 23642060 | AquaMonitor extraction | standard-g, 1 node, 8 GCDs | 20:25 | 836,159 / 836,159 rows (103 chunks); CPU vs GPU 2.1e-5 |
| 23644988 | FIN-Benthic2 folds, first attempt | small, 5 × 8 cores | 45:01 each | TIMEOUT: hung before printing anything, while LUMI logins also hung (~15:25–16:56); cause not confirmed |
| 23645913 | FIN-Benthic2 five folds (code 08d6f39) | small, 5 × 8 cores | 4:48–18:42 | done |
| 23645914 | AquaMonitor five folds | small, 5 × 16 cores | 7:57–23:28 | done |
| 23648776 | output A, montages, code regression, LUMI-vs-Dell, five-fold summary (code 54dffdd) | small, 16 cores | 11:19 | done (first attempt 23645915 failed in the montage step on a single-camera specimen; fixed with a test) |

Use: about 1.9 GPU-h; 114.4 CPU-hours as sacct counts them (logical CPUs), 60 of them in the hung first attempt.

Checks:
- **Code regression:** the earlier capped FIN-Benthic2 fold-0 cohort rerun on LUMI with the new code reproduces every
  reported number of the original run (`docs/evidence/finbenthic2_full_geometry/baseline_regression/`).
- **LUMI vs Dell features (AquaMonitor):** on the 196,983 images of the earlier pilot, max abs diff 4.8e-5, min
  cosine 0.99999999973 (`aquamonitor_full_geometry/extraction/lumi_vs_dell.json`).
- Test rows are removed before any feature lookup and never used. (Correction of my earlier wording: their features
  are read into memory with each cache chunk and then discarded, so "never loaded" was too strong.)

## 3. Results (output B, five folds; fold mean [min, max])

Full table: `docs/evidence/full_geometry_comparison.md`; per fold, per taxon and per pair:
`docs/evidence/*_full_geometry/summary/`.

**One Gaussian vs a 2-group tied mixture (held-out log density per val specimen, class-macro mean).** As computed,
averages include fits our own rule marks unreliable (section 5, finding 1); the right-hand columns use reliable fits
only (`docs/evidence/full_geometry_reliability_audit.json`):

| | FIN-Benthic2 as computed | FIN-Benthic2 reliable only, per fold | AquaMonitor as computed | AquaMonitor reliable only, per fold |
|---|---|---|---|---|
| 8 PCs | −0.057 [−0.169, 0.082]; interval above 0 in 1 fold, below 0 in 1 | +0.117, +0.060, +0.082, 0.000, +0.028 | +0.266 [0.206, 0.328]; above 0 in 5 of 5 | +0.270, +0.321, +0.262, +0.275, +0.261 |
| 16 PCs | −0.075 [−0.205, 0.001]; below 0 in 1 | +0.042, −0.122, +0.029, +0.071, +0.033 | +0.416 [0.323, 0.565]; above 0 in 5 of 5 | +0.506, +0.629, +0.399, +0.518, +0.531 |
| Earlier capped fold-0 pilot (as computed) | −0.094 | | +0.577 | |

"Reliable only" averages over a different set of taxa for each model, so those columns are not like-for-like either.

**What holds under both computations:**
- AquaMonitor's mixture gain persists with all specimens and in all five folds, but is about half the capped pilot's
  (+0.266 vs +0.577 at 8 PCs). The one-Gaussian-per-source-dataset model is better still: +0.455 [0.403, 0.494] at
  8 PCs, interval above 0 in 5 of 5 folds.
- FIN-Benthic2 shows at most a small gain (reliable fits) or none (all fits), far below AquaMonitor's.
- Taxa labelled "credible multiple groups" at 8 PCs: FIN-Benthic2 10.4 [8, 13] (in all 5 folds: *Elmis aenea*,
  *Oulimnius tuberculatus*, *Micrasema gelidum*, *M. setiferum*, Simuliidae); AquaMonitor 25.0 [23, 27] (17 taxa in all
  5 folds). This label is the subject of finding 2.
- Both camera views of a specimen fall in the same group: median 0.83 FIN-Benthic2 vs 0.97 AquaMonitor (8 PCs).
- Metadata association of the 2-group split (train, 8 PCs): AquaMonitor source-dataset ARI ≥ 0.5 in 18.8 [17, 20]
  taxa; FIN-Benthic2 crop-size AUC outside 0.2–0.8 in 15.6 [13, 17] taxa.

**Between taxa (robust across folds):** pairs with val d′ < 2: 1.6 % FIN-Benthic2, 1.8 % AquaMonitor. Closest pairs in
every fold: *Polycentropus flavomaculatus* / *P. irroratus* (mean d′ 0.82), *Baetis rhodani* / *B. vernus* group
(1.47), *Elmis aenea* / *Oulimnius tuberculatus* (1.49); AquaMonitor *Cyrnus trimaculatus* / *Polycentropus
flavomaculatus* (0.68), *Caenis horaria* / *C. luctuosa* (1.21). Rank agreement of pair d′ between folds: Spearman
median 0.91 (FIN-Benthic2), 0.79 (AquaMonitor).

**Output A (exploratory, in-sample: fold-0 train + val pooled, fold-0 test not read):** FIN-Benthic2 7,707
specimens / 38 taxa; AquaMonitor 33,732 / 42. In-sample BIC prefers the 2-group tied model in 23 of 38 FIN-Benthic2
taxa and 40 of 42 AquaMonitor taxa (8 PCs); both views in the same group: 0.81 vs 0.97. Not held-out evidence.

## 4. Limits

- Five folds share specimens: fold ranges show consistency across assignments, not independent replications.
- Intervals are conditional on the fitted PCA and models and on the included taxa; no familywise error control.
- Mixture gains can reflect skew or heavy tails, not separate groups; ARI and AUC show partition agreement or rank
  separation, not cause. Crop area is not specimen size; life stage is not annotated.
- FIN-Benthic2 has no source field, so the source model cannot be fitted there.

## 5. Methods review by Codex Astra (high effort, read-only)

**What was asked.** After the code for this task was written, the author asked for an Astra review of the
mathematical and statistical methods (not the results, not the code style). Fresh session; handed over:
`task.md`, this file, the geometry, mixture, specimen-mean, frame-selection, metadata, summary and output-A code, and
a context note (`Feature_geometry_data/full_specimen/astra_methods_review/context.md`, outside Git). Nine questions:
specimen unit; global whitened PCA and held-out density comparison (regularisation, starts, eligibility, d+1 rule);
bootstrap and class-macro interval; verdict rule and split stability; source model; d′, radius separation, subspace
overlap and effective rank with unequal class sizes; five-fold aggregation; output A; metadata measures.
Record: `~/agent_consults/2026-11-11_183354_codex_gpt-6-astra_high.md` (session `01a1cb75-723b-7eb0-ba3a-10db70d181be`).

**Astra's verdict:** "needs changes — the core density comparison is sound, but reliability filtering, fixed-size
summaries, and interpretation of the verdict need correction." Judged correct: specimen embedding, train-only PCA,
float64 fits, parameter counts, paired bootstrap, stratified class-macro interval, source model, geometry formulas,
fold aggregation, in-sample labelling of output A.

**Findings, my check in the code, and their effect on the saved outputs** (`scripts/audit_mixture_reliability.py`):

| # | Astra finding | Confirmed in code | Effect in this run |
|---|---|---|---|
| 1 (high) | Averages include fits marked unreliable (a group with < d+1 specimens), while verdicts exclude them | yes: scores of unreliable fits enter the bootstrap and macro mean (`class_geometry_pilot.py`, `fit_class` / `summarise`) | large for FIN-Benthic2 (section 3, e.g. fold 0 at 8 PCs −0.069 → +0.117; 3-group tied −1.589 → +0.242), small for AquaMonitor's 2-group model. Also affects the earlier capped pilots |
| 2 (high) | "Credible" passes if *any* mixture beats one Gaussian while stability is checked for the 2-group tied split only; a mixture gain may be skew, not groups; no multiplicity control | yes | at 8 PCs, 1–3 FIN-Benthic2 and 1–6 AquaMonitor credible taxa per fold have no 2-group-tied interval above 0 (16 PCs: 0–3 and 0–2). Stability refits never failed in any fold (that sub-point has no effect here) |
| 3 (medium) | "n30" summaries use fewer than 30 specimens for small taxa | yes (`specimen_geometry.py`, `within_class`) | 0–2 FIN-Benthic2 taxa per fold; none in AquaMonitor |
| 4 (medium) | Source-model eligibility uses the one-Gaussian parameter count | yes | none: no source fit exceeds its true budget |
| 5 (medium) | Five-fold summary drops the taxon count behind each average; missing verdicts count as non-credible | yes (`summarise_five_folds.py`) | denominators: FIN-Benthic2 34–36 taxa per fold, AquaMonitor 40–41 |
| 6 (medium) | ARI near 0 / AUC near 0.5 described as "unrelated"; five-fold metadata counts are train-only but not labelled | yes (`component_metadata.py` docstring, `summarise_five_folds.py`) | wording only |
| — | "Features never loaded" for test rows too strong | yes (`feature_lookup.py` loads whole caches) | wording only; no leakage |

**What needs to change (Astra's proposals, not applied):**
1. Primary averages over reliable fits only (both the mixture and one Gaussian), with the taxon count per
   comparison; all-fit averages kept as a labelled diagnostic. Add the same reliability flag to output A.
2. Per-taxon label: require the 2-group tied model itself to beat one Gaussian and its split to be stable, and rename
   it to an exploratory "mixture-supported split"; report other mixtures separately; state intervals as pointwise.
3. n = 30 summaries only for taxa with at least 30 specimens, with the count of contributing taxa.
4. Source-model eligibility with its own parameter count; document its pooled-group rule.
5. Carry taxon and pair counts through the five-fold summary; mark unavailable verdicts explicitly.
6. Metadata wording ("partition agreement", "rank separation"); label train vs val association counts.

## 6. Decisions for the author

- **D1 (finding 1):** dataset-level averages over reliable fits only (recommended), or all fits as now.
- **D2 (finding 2):** per-taxon label from the same model (2-group tied gain + its stability, renamed exploratory;
  recommended), or keep the any-mixture rule and only rename, or keep as is.
- **D3 (findings 3–6):** apply the smaller fixes with regression tests and rerun the CPU stage on LUMI (about
  1 h; extraction does not need to be repeated).

## 7. Code and tests

New since the preflight: `scripts/descriptive_geometry.py`, `scripts/summarise_five_folds.py`,
`scripts/compare_caches_by_name.py`, `scripts/audit_mixture_reliability.py`, `src/statistics/specimen_geometry.py`,
`scripts/lumi/geometry_{folds,inputs,summary}.sh`, `tests/test_class_geometry_montage.py`. Changed: `--fold` in the
geometry, metadata and montage scripts (test rows dropped before lookup); chunked camera re-identification;
specimen-mean helpers take the split column; montage handles single-camera specimens. Full suite: 61 passed.

## 8. Rerun

```bash
# Dell (metadata only)
python scripts/build_full_specimen_selection.py --dataset finbenthic2 --csv CSV --hdf5 H5 --ida IDA --out DIR --reuse FOLD0_COHORT
python scripts/build_full_specimen_selection.py --dataset aquamonitor --manifest MANIFEST --out DIR
# LUMI (code dir C, results O)
sbatch scripts/lumi/extract_dinov3_l.sh C MANIFEST HDF5 OUT          # per dataset
sbatch --array=0-4 [--cpus-per-task 8 --mem 32G --time 00:45:00] scripts/lumi/geometry_folds.sh C finbenthic2 O
sbatch --array=0-4 scripts/lumi/geometry_folds.sh C aquamonitor O
sbatch scripts/lumi/geometry_summary.sh C O
```
