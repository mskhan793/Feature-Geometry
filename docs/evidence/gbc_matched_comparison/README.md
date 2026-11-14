# Evidence — paper-faithful GBC and matched PCA-vs-FDA comparison (task.md e4d291d)

40 encoder × dataset combinations (80 runs: FIN-Benthic2 and AquaMonitor five folds each, six single-partition
datasets), train + validation only. Nothing was refitted: every number is a sum or summary of the per-pair
Bhattacharyya values saved by the geometry replication's `fda` analysis (`scripts/pca_fda_gbc.py`, LUMI
`/scratch/project_462001534/feature_geometry/replication/runs/<encoder>/<dataset>/<partition>/fda/pairs_<arm>.csv`).

## Protocol

**Authors' approach (Pándy et al., CVPR 2022; `~/Documents/Research/methods/gbc_pandy2022/paper.pdf`).**
- Per class c, a Gaussian N(μ_c, Σ_c) of the target embeddings.
- Bhattacharyya distance (Eq. 1):
  D_B = (1/8)(μ_i − μ_j)ᵀ Σ⁻¹ (μ_i − μ_j) + (1/2) ln( det Σ / sqrt(det Σ_i det Σ_j) ), with Σ = (Σ_i + Σ_j)/2.
- BC = exp(−D_B). GBC = −Σ over ordered distinct class pairs of BC (Eq. 3). Larger (less negative) GBC means less
  estimated overlap.
- For image classification the paper uses unwhitened PCA to 64 dimensions and spherical covariances, computed on the
  target training set. This is a one-score, training-only use; the paper has no validation re-estimate.
- **Paper form here:** spherical Σ_c = σ_c² I. σ_c² is the mean per-dimension unbiased (n − 1) variance, and the
  log-determinant term carries the factor d (Eq. 2).
- **Code-release form** (google-research `stable_transfer`, labelled reproduction): no factor d in the
  log-determinant term, population variance, 1e-4 variance floor (`docs/bhattacharyya_source_note.md`).

**Our extension (not from the paper; `docs/pca_fda_gbc_method_note.md`).**
- FDA projection fitted on train: S_B v = λ S_W* v, where S_W* is the Ledoit–Wolf-shrunk within-class scatter.
  Orthonormal basis of the top d directions, centred on the train mean, d = 8, 16, 32 (FDA rank permitted every d in
  every run).
- Same spherical BC and GBC as the paper form.
- Raw generalised eigenvectors and halved / doubled shrinkage are sensitivity arms only.
- **Our train/val diagnostic:** in the same train-fitted projection, class Gaussians are fitted separately on the
  train units and on the val units, and the pair BCs of both are compared. The projection never sees val.

**Arms and conventions** (`src/statistics/gbc_matched.py`).
- **A:** `pca64`, paper form and code form, training only, all train-valid pairs.
- **B:** `fda{d}` (primary) and `fda{d}_{raw,shrink0.5,shrink2}` (sensitivity).
- **C:** `pca{d}` vs `fda{d}` at the same d. Every quantity is computed over the intersection of both arms' common
  pairs, identified by class pair ids. Different class sets would make a comparison ineligible; none occurred.
  PCA-64 is never compared with an FDA arm.
- **D:** GBC_train and GBC_val are both summed over P_common, the pairs valid on train and on val. The train-only sum
  over all train-valid pairs is kept as auxiliary `gbc_train_all`. The earlier summaries summed val over P_common but
  train over all train-valid pairs; `previous_ratio` in `arms_long.csv` reproduces that convention.
- **Unordered pairs:** GBC = −2 Σ BC, equal to the paper's ordered sum (tested).
- **E — ratio:** val/train = mean BC val / mean BC train over P_common. It is reported only when the train mean BC is
  ≥ 1e-6, a predeclared threshold that was never reached. A ratio > 1 means a larger estimated overlap magnitude on
  val, not classification error. Mean BC differences, the per-pair ΔBC = BC_val − BC_train quantiles, the share of
  pairs with ΔBC > 0, Spearman(train distance, val distance) and retention of the 20 hardest pairs are always reported.
- **E — mean/covariance terms:** reported for the spherical form (class-specific σ², so the log-det term is defined
  and generally non-zero) and for the full form.
- **F — support:** classes need ≥ 15 train and ≥ 5 val units (as in the replication). Full covariances need ≥ 2 d
  units per class on the respective split. Low validation support is examined through strata of min(n_val) of a pair
  (5–9, 10–19, ≥ 20). Folds are shown individually. Their val sets overlap, so no fold-independence or pair-independence
  interval is claimed.

## Checks

- **Leakage gate** (every run, `scripts/gbc_matched_comparison.py`): the GBC classes equal the cohort's eligible
  classes, and per-class train/val unit counts equal the train/val cohort built after dropping the partition's test
  rows (`../geometry_replication/inputs/support/`). The dropped test row count is recorded.
- **Regression** (every run): the recomputed `gbc_train_all` and `gbc_val_common` equal the saved summaries. The
  maximum relative difference is in `summary/runs.csv`.
- **Bounded verification before scaling:** `verification/` holds the DINOv3-L FIN-Benthic2 fold0 and Oxford-IIIT Pet
  runs computed on the Dell. They are identical to the LUMI outputs, apart from the source path.
- **Tests:** `tests/test_gbc_matched.py`, covering (a) agreement with the earlier summary code, (b) the
  ordered/unordered factor 2, (c) identical pair sets across train/val and across matched arms, (d) the leakage gate,
  (e) near-zero GBC.

## Runs and outputs

- **LUMI:** array job 23725233 (small partition, 80 tasks, project 462001824, code 3cbc101,
  `scripts/lumi/gbc_matched.sh`). Outputs are in `/scratch/project_462001534/feature_geometry/gbc_matched/out` and
  locally in `~/Documents/Research/Feature_geometry_data/gbc_matched/lumi_out`. Each run has `matched.json`,
  `per_class.csv` (per class: mean and max BC on train and val, primary arms, spherical) and `hardest_pairs.csv`.
- **`summary/`** (`scripts/gbc_matched_summary.py`):
  - `tables.md` holds the dataset × encoder tables and counts;
  - `runs.csv`, `arms_long.csv`, `matched_long.csv`, `strata_long.csv` hold the long tables;
  - `per_fold.csv` holds the per-partition spherical ratios.
- **`accounting/`:** `sacct_23725233.txt` and `resources.json`.

## Limitations

- **Unequal sample sizes:** the train and val Gaussians come from different numbers of units (val classes are smaller;
  per-class counts in each run's `classes.csv`), so the variance and mean estimates on val are noisier. The direction
  of the resulting bias in BC is not derived here.
- **Projection selection effect:** the projection is fitted on train. FDA uses the train labels, so its train
  separation includes a selection effect that val does not share. The ratio alone does not establish overfitting.
- **High-BC pairs dominate:** a GBC sum is driven by its high-BC pairs, so a shift of the sum can come from a minority
  of pairs. The per-pair ΔBC quantiles are reported for this reason (`arms_long.csv`, `delta_bc_p10` / `_median` /
  `_p90`).
- **Not accuracy:** these are estimates of Gaussian overlap, not classification accuracy or transferability. No
  model-selection hypothesis is tested.
