# Evidence — class-balanced and taxonomy-informed covariance pooling (task.md f54f534)

- **Provenance, taxonomy feasibility and the predeclared design:** `docs/balanced_taxonomic_pooling_note.md`.
- **Code:** commit `ff2cdf3`.
  - Experiment: `scripts/balanced_taxonomic_pooling.py`, `src/methods/taxonomic_pooling.py`,
    `src/data/taxonomy.py`, `src/statistics/heldout_density.py`.
  - Summary: `scripts/summarise_balanced_taxonomic.py`, `src/statistics/fold_summary.py`.
  - Audit: `scripts/audit_taxonomy.py`, `scripts/make_finbenthic2_taxonomy.py`, `scripts/compare_taxonomy_with_v1.py`.
  - LUMI job: `scripts/lumi/balanced_taxonomic_pooling.sh`.
- **LUMI** (project 462001534, `small`, 16 cores / 64 GB / 1 h limit per task):
  - preliminary task `23686150` (AquaMonitor fold0), checked before the rest;
  - `23686221` (the other 9 dataset-folds);
  - all COMPLETED with empty stderr; 14.32 allocated CPU-hours (`sacct.txt`); logs in `logs/`.
- **Outputs:**
  - LUMI `/scratch/project_462001534/feature_geometry/balanced_taxonomic_pooling/<dataset>/<fold>/`;
  - with per-specimen log-likelihoods in `~/Documents/Research/Feature_geometry_data/balanced_taxonomic_pooling/`.

## Files

- `taxonomy_audit/`:
  - per-dataset `classes.csv` (ancestry, rank, specimens) and `donors.csv` (per fold, dimension and target: donor
    level, donors, df, genus / family mates);
  - `summary.json`;
  - the transcribed FIN-Benthic2 table;
  - `v1_crosscheck.{csv,json}`.
- `runs/<dataset>/<fold>/`: `summary.json` (levels, failures, contrasts by group) and `perclass.csv` (per class and
  space: level, donors, nearest donors, Jaccard, β, α, identity scales, conditioning, status, held-out
  log-likelihood of every arm, paired contrasts with intervals and parts).
- `summary/<Dataset>/`:
  - `folds.csv` and `groups.csv` (strata, n_c ≤ d, taxonomy used / genus / family / fallback);
  - `levels.csv`, `paths.csv`;
  - `reproduction.json` (exact match of LW, coupled and `pool_spec` with the earlier runs);
  - `resources.json`;
  - `summary/summary.md` (all tables).
