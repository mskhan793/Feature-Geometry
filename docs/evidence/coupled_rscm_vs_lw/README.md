# Evidence — per-class Ledoit–Wolf vs coupled RSCM (task.md 61ceb61)

- **Method and provenance:** `docs/coupled_rscm_source_note.md` (paper, authors' code `bb09e12`, paper-vs-code check,
  the one adaptation, parameter conventions).
- **Code:** commit `8cf3738` (`scripts/coupled_rscm_vs_lw.py`, `src/methods/coupled_rscm.py`,
  `third_party/coupled_rscm/`, `scripts/lumi/coupled_rscm_vs_lw.sh`); summary `scripts/summarise_coupled_rscm.py`
  (run on the Dell; examples_by_fold added after the LUMI runs, read-only on their outputs).
- **Inputs:** existing DINOv3-L full-specimen caches and selections (`scripts/lumi/geometry_inputs.sh`), FIN-Benthic2
  derived five-split protocol and AquaMonitor Classification official folds; train fits, val scores, test rows dropped
  before the feature lookup. PCA fitted on each fold's train specimens; 16 and 64 dimensions.
- **LUMI jobs** (project 462001534, partition `small`, 16 cores / 64 GB / 30 min limit per task):
  `23676999` (array 0 = FIN-Benthic2 fold0, the real-fold test, run first and checked) and `23677160` (array 1–9);
  all COMPLETED, 10–50 s each, 9,408 allocated CPU-seconds in total (2.61 CPU-hours). Logs in `logs/`.
- **Outputs on LUMI:** `/scratch/project_462001534/feature_geometry/coupled_rscm_vs_lw/<dataset>/<fold>/`; copy with
  per-specimen log-likelihoods (`perspecimen.csv`) in `~/Documents/Research/Feature_geometry_data/coupled_rscm_vs_lw/`.

## Files

- `runs/<dataset>/<fold>/summary.json` — eligibility, excluded classes, the 4-class original-vs-adapted check, α/β
  ranges, numerical failures, macro/micro paired differences with bootstrap intervals, strata.
- `runs/<dataset>/<fold>/perclass.csv` — per class and dimension: train/val counts, stratum, pooled weight π_k, LW α
  (shrinkage), coupled α (kept) and 1 − α, β, η/γ/κ, symmetry / minimum eigenvalue / condition number of both
  covariances, status of every estimator, mean held-out log-likelihood of LW, coupled and the three labelled corners
  (pooled SCM, class SCM, class scaled identity), paired difference mean / median / bootstrap 95 % interval.
- `summary/<Dataset>/` — `folds.csv`, `strata.csv`, `parameters.csv`, `reproduction.json` (LW reference vs the
  earlier `lw_fit` run), `examples.csv`, `examples_by_fold.csv`; `summary/summary.md` all tables.
