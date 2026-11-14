# Evidence — geometry replication over 8 datasets × 5 frozen encoders (task.md a82e89e + I-JEPA amendment 3b897b0)

Replication of the established frozen-feature geometry analyses (FIN-Benthic2 / AquaMonitor DINOv3-L tasks 92a605a –
f54f534) on all 40 encoder × dataset train+val caches. No classifier, no accuracy, no test data, no new formulations.

## Inputs

- **Caches:** the 32 C3-checked caches of task d57a630 (`../benchmark_readiness/c2_plan/records.tsv`) and the eight
  I-JEPA ViT-H/14 caches (`../benchmark_readiness/ijepa_plan/records.tsv`, records `../benchmark_readiness/ijepa_records/`).
- **Cohorts** (`inputs/`, `scripts/replication/build_cohorts.py`, `src/data/benchmark_cohort.py`):
  - FIN-Benthic2 / AquaMonitor: the full-specimen selections unchanged (partitions fold0..fold4; each fold's test
    rows are dropped before any feature lookup);
  - BIOSCAN-5M, DF24, Food-101, Oxford-IIIT Pet, CIFAR-100, FGVC-Aircraft: the train+val extraction manifests in the
    same schema (partition ``official``; no test row exists in them, asserted). Unit = ``unit_id`` of the standardised
    manifest (BIOSCAN record, DF24 observation, exact-content hash of an image elsewhere); unit embedding = mean of its
    images; one constant camera / acquisition run.
  - Exact-duplicate content units whose images carry different official labels are kept, one unit per (unit, label)
    (``inputs/support_summary.json``: ``conflicting_label_units``). Nothing is removed or relabelled.
- **Eligibility** (unchanged established rule): a class is analysed if it has >= 15 train and >= 5 val units
  (``inputs/support/<dataset>_<partition>.csv``, ``inputs/support_summary.json``). Model-specific rules (parameters
  <= n d / 2 for mixtures, >= 2 d units for full covariances, n > d for sample covariances) are unchanged.

## Analyses and runs (`scripts/replication/make_tasks.py`, `scripts/lumi/replicate_geometry.sh`)

One LUMI CPU array task per encoder × dataset × partition × analysis (task tables in `tasks/`):
`geometry` (class_geometry_pilot.py + component_metadata.py), `overlap` (bhattacharyya_overlap.py), `fda`
(pca_fda_gbc.py), `lw` (lw_fit_diagnostics.py), `coupled` (coupled_rscm_vs_lw.py), `pooling`
(balanced_taxonomic_pooling.py ``--no-taxonomy``). Outputs on LUMI:
`/scratch/project_462001534/feature_geometry/replication/runs/<encoder>/<dataset>/<partition>/<analysis>/`.

Batches (task tables in `tasks/`; job ids, code commits and billed project per run in
`final/completion/run_provenance.csv`, parsed from the Slurm logs):

| Batch | Task table | Array job | Tasks | Project |
|---|---|---|---|---|
| P1 reproduction (DINOv3-L FIN-Benthic2 / AquaMonitor) | `p1_reproduction.tsv` | 23710715 | 60 | 462001534 |
| P2 pilot FIN-Benthic2 × SigLIP 2 | `p2_fin_siglip2.tsv` | 23710716 | 30 | 462001534 |
| P2 pilot CIFAR-100 | `p2_cifar.tsv` | 23710717 | 12 | 462001534 |
| B1 five-fold | `b1_five_fold.tsv` | 23711050 | 150 | 462001534 |
| B1 general images | `b1_general.tsv` | 23711310 | 84 | 462001534 |
| B2 probes (BIOSCAN-5M / DF24, light) | `b2_probe_light.tsv` | 23711344 | 8 (5 done; 1 failed, rerun in B2a; 2 cancelled with author approval, rerun in B2d) | 462001534 |
| B2 probes (covariance) | `b2_probe_cov.tsv` | 23711345 | 4 | 462001534 |
| B3 I-JEPA (all except BIOSCAN-5M / DF24 covariance) | `b3_ijepa_light.tsv` | 23713036 | 84 | 462001534 |
| B2a BIOSCAN-5M / DF24 light | `b2a_light.tsv` | 23713184 | 25 | 462001824 |
| B2b BIOSCAN-5M / DF24 coupled + pooling | `b2b_cov.tsv` | 23713335 | 16 | 462001824 |
| B2c BIOSCAN-5M / DF24 LW | `b2c_lw.tsv` | 23713491 | 8 | 462001824 |
| B2d probe LW rerun (faster, byte-identical code) | `b2d_probe_lw_rerun.tsv` | 23715771 | 2 | 462001824 |

`all_tasks.tsv` (384 rows, four encoders) + `b3_ijepa_all.tsv` (96 rows, I-JEPA) = the 480 tasks of the matrix.
Each output was written by exactly one run (asserted); runs that found a DONE marker skipped.

## Final outputs (`final/`)

- `summary/` (`scripts/replication/summarise_replication.py`): `task_status.csv` (480 done), `indicators_long.csv`
  (every indicator per run, with intervals), `indicators_by_combination.csv` (per encoder × dataset over partitions:
  mean, min, max, partitions with interval above / below 0), `completion_by_analysis.csv`.
- `separation/` (`scripts/replication/separation_table.py`): d' per run and per combination.
- `completion/` (`scripts/replication/completion_matrix.py`): `completion_matrix.csv` (the 40-row matrix: units,
  support rule, classes, eligible pairs, native space and dimension, class-fit and pair-score status counts, contrast
  numerical failures, failed arms, jobs, code, project), `arm_status.csv` (every arm × setting × partition with its
  status counts), `run_provenance.csv` (483 runs).
- `headline/` (`scripts/replication/headline_tables.py`): one dataset × encoder table per established headline
  indicator plus cross-table counts.
- Per-class and per-fold outputs stay in the run folders (LUMI, path above; local copy without per-specimen tables
  at `~/Documents/Research/Feature_geometry_data/replication/runs`).

## Numerical failures

- `coupled_no_pooling` (mechanism-decomposition arm of the coupled study, sample covariance without pooling):
  9,203 class fits numerically singular, mostly in native spaces and at 64 PCs; the two decomposition contrasts that
  use it (`pooling_at_coupled_shrink`, `shrink_at_no_pooling`) record 9,203 failures each.
- Primary `coupled` arm: one class per fold, DINOv3-L × FIN-Benthic2, native space (coupled and pooling scripts);
  the earlier run had the same failure (one class per fold) — 15 contrast failures in total.
- No other arm failed. Non-applicable or not-run arms are listed with their status in `arm_status.csv`.

## Resources (`accounting/`, allocation-based, `scripts/benchmark/sum_resources.py`)

`sacct_final.txt` + `sacct_final_ijepa_extraction.txt` -> `resources_final.json`: 5.01 GPU-h (I-JEPA extraction and
gates), 5,400.37 billed CPU core-h (geometry replication 5,374.92); per project 462001534 1,662.83 core-h + 5.0
GPU-h, 462001824 3,737.55 core-h. `allocations_final.txt`: balances on 2026-11-14. The 21:13 snapshot is kept.

## Reproduction of the earlier DINOv3-L results (`reference_check/`)

`scripts/replication/compare_reference.py`: every CSV / JSON of the earlier evidence folders compared cell by cell
with the new DINOv3-L FIN-Benthic2 / AquaMonitor runs (predeclared relative tolerance 1e-9; timing / path entries
ignored). AquaMonitor: all 26 compared run outputs (every earlier output; overlap existed for fold0 only) identical within
tolerance. FIN-Benthic2: 62 mismatches, all in the geometry
outputs — mixture component labels swapped (0 <-> 1; the same partition), montage specimens chosen from the swapped
component, and one stability statistic (fold4, one class: G2_tied split ARI median 0.0034 vs 0.0028); every other value
within 2.4e-10 relative. Most plausible cause: rounding-level input differences flipping a component order; not
traced further.

## Deviations from the earlier runs (forced by data or author instruction)

1. Taxonomy-informed pooling (pool_tax) and its nearest-mean control (pool_near, whose donor count is set by the
   taxonomic donors) are not run: author instruction in chat, 2026-11-13 ("for now we don't consider the taxonomy").
   The other pooling arms are computed exactly as before.
2. Native spaces: DINOv2-L, DINOv3-L and I-JEPA features have rank p - 1 (final-LayerNorm null direction), so the
   established ``native`` / ``native_alt`` (one coordinate dropped) are used. BioCLIP 2 and SigLIP 2 features are full
   rank: ``native_all`` (all coordinates) and ``native`` (last coordinate dropped, sensitivity).
3. Datasets without camera or acquisition fields: camera re-identification, two-view agreement and the source-group
   model are not applicable (NaN / not fitted); component metadata ARIs only for AquaMonitor (dataset, year, site)
   and FIN-Benthic2 (log image area).
4. Not replicated: descriptive_geometry.py (pools fold0 train and val, in-sample; conflicts with train-only fitting)
   and bhattacharyya_orientation_summary.py (post-hoc summary of one earlier run).
5. Code changes with identical results: per-class principal axes computed once (specimen_geometry.between_classes)
   and per-class statistics computed once in lw_fit_diagnostics.py pair loops (byte-identical outputs checked); an
   overlap figure no longer fails when no class pair has a full covariance; unit ids read as text in
   component_metadata.py (DF24 observation ids are numeric strings).
