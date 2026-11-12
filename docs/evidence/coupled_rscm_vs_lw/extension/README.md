# Evidence — native space and mechanism decomposition (author's approval 2026-11-12; response.md §8)

- Code `41cbfdf`; LUMI jobs `23683103` (timed AquaMonitor fold0) and `23683215` (other 9 dataset-folds), output
  `/scratch/project_462001534/feature_geometry/coupled_rscm_vs_lw_v3/`; per-specimen files in
  `~/Documents/Research/Feature_geometry_data/coupled_rscm_vs_lw_v3/`. Superseded: `23679633` (code `ea57ebd`,
  all 1024 coordinates; singular covariances), output `coupled_rscm_vs_lw_v2/` on LUMI.
- `summary/<Dataset>/`: folds.csv (per fold x space x contrast), groups.csv (strata and n_c <= d), parameters.csv,
  corners.csv, examples*.csv, reproduction.json (LW vs lw_fit), resources.json; `summary/summary.md` all tables.
- `runs/<dataset>/<fold>/`: summary.json (incl. full 1024-coordinate covariance rank, numerical audit) and perclass.csv.
- `regression_vs_first_run.json`: PCA results equal the first run exactly (scripts/check_coupled_regression.py).
- `sacct.txt`: elapsed, CPU time and memory of all three jobs.
- `../native_null_space/`: LayerNorm constraint audit on local DINOv3-L features (scripts/audit_feature_null_space.py).
