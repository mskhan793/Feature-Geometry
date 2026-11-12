# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 61ceb61, "Focused covariance study: per-class Ledoit–Wolf versus Coupled RSCM".
**Status: done for all five folds of FIN-Benthic2 and AquaMonitor; stopped for the author's review.** Frozen DINOv3-L,
existing specimen embeddings; train specimens fit PCA, class means and both covariance estimators; val specimens give
the held-out log-likelihood; no test rows, no classifiers, no other methods, nothing tuned on validation.

Previous response (PCA vs FDA GBC, Ledoit–Wolf diagnostics §8, shrinkage literature §9): commit ea56c64.

## Verdict

*(Revised after the Astra high methods review, §7; first version in commit e3d6e1e.)*

**No universal winner.** The coupled estimator (CRSCM) **loses at 16 PCs** in both datasets (class-macro interval
below 0 in 3 FIN-Benthic2 and all 5 AquaMonitor folds). **At 64 PCs it improves the rare-class averages** (taxa with
15–63 train specimens: mean +2.97 FIN-Benthic2, 20 of 25 class-folds better; +1.10 AquaMonitor, 28 of 39; interval
above 0 in 4 of 5 folds in both) **and FIN-Benthic2's primary class-macro outcome** (positive in all 5 folds, interval
above 0 in 3); AquaMonitor's 64-PC macro is mixed (interval above 0 in 1 fold). Classes with ≥ 64 train specimens lose
slightly on average. **Per specimen (micro) there is no clear gain:** no micro interval lies above 0 (two positive
point estimates), and 15 of the 20 fold × dimension intervals lie below 0. Keeping per-class LW as the main class
model is a reasonable operational choice, not an established general superiority. Both estimators were designed for
dimension comparable to or above the class size; a bounded native-1024 comparison is proposed in §7 (not run).

## 1. Sources and fidelity (gate passed)

- **Paper** opened and read locally: Raninen & Ollila, IEEE TSP 69:5681–5692, 2021, doi 10.1109/TSP.2021.3118546
  (`~/Documents/Research/methods/Shrinkage/Aalto_thesis/…Multiple_Classes.pdf`, SHA256 `14853bad…`). Ledoit & Wolf
  (2004) preprint retained (`…/ledoit_wolf2004/paper.pdf`, `08e79244…`). Full record: `docs/coupled_rscm_source_note.md`.
- **Authors' code** `github.com/EliasRaninen/CoupledRSCM` @ `bb09e12` (MIT), `Python/multiclass_rscm.py`, vendored
  **byte-identical** in `third_party/coupled_rscm/` (test checks the SHA256). Used directly: `estimate_parameters` →
  `rscmpool(averaging=False)`, i.e. class-specific (α_k, β_k) from the authors' plug-in estimates (elliptical
  kurtosis, spatial-sign sphericity, SSCM inner products) and their grid + alternating search.
- **Paper vs code**, checked line by line: estimator, unbiased SCM, π_k = n_k/N pooling, all nine MSE-polynomial
  coefficients (Appendix A) and the plug-in formulas agree, including the clip of γ̂_k to [1, p] (paper Sect. IV-B;
  the first version of this note wrongly said only the code clips). Its slow-convergence warning can never fire, and
  its stopping rule adds the α and β changes before the norm, so they can cancel (convergence not audited here); it
  imports `scipy.stats.stats.kurtosis` (works in SciPy 1.18.0,
  deprecated).
- **One essential adaptation:** `rscmpool` evaluates its grid with `.reshape(4)`, so the original works only for
  exactly 4 classes (it raises `ValueError` otherwise — test). We replace it with `.reshape(K)` in memory at load time
  (`src/methods/coupled_rscm.py`; the file is not edited). **Equivalence:** on 4-class fixtures and, in every LUMI
  run, on the first four eligible classes of each real fold and dimension, the unmodified and adapted code give
  identical α, β and Σ̂ — maximum absolute difference **0.0** in all 20 checks.
- **Conventions:** the coupled α_k is the weight *kept* on T_k(β_k); 1 − α_k is its shrinkage toward the scaled
  identity. LW's α is the shrinkage itself. They are reported side by side (`coupled_shrink_to_identity`,
  `lw_alpha_shrink`), never as the same quantity. The coupled estimator starts from the unbiased S_k (n − 1), LW from
  the biased one (n); each as defined by its source.
- **LW reference reproduced exactly:** against the earlier `lw_fit_diagnostics.py` outputs (section 8 D3), all 350
  (FIN-Benthic2) and 408 (AquaMonitor) class × fold × dimension rows match, maximum difference in held-out
  log-likelihood and α **0.0**, identical train/val counts. The LW formula is now one shared function
  (`lw_class_covariance`) used by both scripts.

## 2. What ran

- Code `8cf3738` (`scripts/coupled_rscm_vs_lw.py`, `scripts/summarise_coupled_rscm.py`,
  `scripts/lumi/coupled_rscm_vs_lw.sh`, tests `tests/test_coupled_rscm.py`; 91 tests pass). Summary tables extended
  afterwards (corner table, examples per fold) on the saved outputs; same results.
- Protocol identical to section 8 D3: specimen = mean of camera means over all selected frames; PCA on each fold's
  train specimens, 16 and 64 dimensions; class mean = train sample mean; eligible = ≥ 15 train and ≥ 5 val specimens
  (34–36 FIN-Benthic2 and 40–41 AquaMonitor classes per fold); both estimators on exactly the same classes,
  specimens and means; the coupled pooled matrix is over the eligible classes. Train and val specimens disjoint and
  unique (asserted); test rows dropped before the feature lookup.
- **LUMI:** real-fold test first — job `23676999` (FIN-Benthic2 fold0; checked: 4-class equivalence 0.0, LW reference
  identical), then `23677160` (the other 9 dataset-folds). Partition `small`, 16 cores / 64 GB / 30 min limit; all
  COMPLETED in 10–50 s; **2.61 CPU-hours** allocated in total; project 462001534. No GPU.
- Evidence: `docs/evidence/coupled_rscm_vs_lw/` (per-fold `summary.json` + `perclass.csv`, `summary/<Dataset>/`
  tables, logs, README). Per-specimen log-likelihoods outside Git:
  `~/Documents/Research/Feature_geometry_data/coupled_rscm_vs_lw/`.

**Measurement:** held-out Gaussian log-likelihood of each val specimen under its own class's N(μ_c, Σ̂_c); paired
difference coupled − LW per specimen (positive = coupled better). Class-macro (primary) = mean over classes of the
class mean; specimen-micro (secondary). 95 % intervals: stratified bootstrap over val specimens (2,000 draws),
conditional on the train fits. Raw log-likelihoods are not compared across dimensions or datasets.

## 3. Results (paired difference coupled − LW; fold mean [min, max] over five correlated folds)

| | FIN-Benthic2 16 PCs | FIN-Benthic2 64 PCs | AquaMonitor 16 PCs | AquaMonitor 64 PCs |
|---|---|---|---|---|
| Class-macro | −0.0562 [−0.1231, −0.0055] | +0.3569 [+0.0169, +0.7632] | −0.1480 [−0.1735, −0.1115] | +0.0503 [−0.1561, +0.3465] |
| folds with interval above / below 0 | 0 / 3 | 3 / 0 | 0 / 5 | 1 / 0 |
| Specimen-micro | −0.0434 [−0.0568, −0.0307] | −0.0440 [−0.1113, +0.1084] | −0.0247 [−0.0279, −0.0180] | −0.0378 [−0.0647, +0.0064] |
| folds with interval above / below 0 | 0 / 5 | 0 / 2 | 0 / 5 | 0 / 3 |
| Numerical failures | 0 | 0 | 0 | 0 |

Per fold (classes, macro and micro with intervals, wins): `summary/<Dataset>/folds.csv`.

**By train-size stratum** (class-folds; mean / median class difference; class-folds where coupled is better; folds
whose stratum interval is above / below 0):

| Stratum | FIN-Benthic2 16 PCs | FIN-Benthic2 64 PCs | AquaMonitor 16 PCs | AquaMonitor 64 PCs |
|---|---|---|---|---|
| 15–63 | 25 (7 taxa); −0.0776 / −0.1707; 8 better; 0 / 1 | 25; **+2.9660 / +3.0118; 20 better; 4 / 0** | 39 (9 taxa); −0.2616 / −0.2245; 10 better; 0 / 3 | 39; **+1.0955 / +0.5477; 28 better; 4 / 0** |
| 64–255 | 119 (26 taxa); −0.0559 / −0.0415; 32 better; 0 / 4 | 119; −0.0410 / −0.0572; 55 better; 0 / 0 | 95 (19 taxa); −0.1888 / −0.1635; 13 better; 0 / 5 | 95; −0.2660 / −0.1856; 36 better; 0 / 2 |
| ≥ 256 | 31 (7 taxa); −0.0381 / −0.0336; 9 better; 0 / 4 | 31; −0.1959 / −0.1823; 3 better; 0 / 4 | 70 (14 taxa); −0.0291 / −0.0042; 20 better; 0 / 5 | 70; −0.1018 / −0.0220; 15 better; 0 / 2 |

- **Rare taxa gain only at 64 PCs**, where 15–63 train specimens give a class covariance of rank ≤ n − 1 < 64
  (singular). Borrowing the pooled covariance *can* supply missing directions, but this comparison does not separate
  that from the different identity shrinkage, target scale or plug-in estimation. At 16 PCs the same taxa lose
  slightly; there, most rare classes have n − 1 ≥ 16, but classes with 15 or 16 train specimens are still singular
  (corrected: the first version said n > d for every eligible class).
- **Larger classes lose slightly**, most clearly ≥ 256 at 64 PCs in FIN-Benthic2 (3 of 31 better, 4 folds below 0)
  and every stratum at 16 PCs in AquaMonitor (5 folds below 0). A large n does not by itself show that a covariance is
  well estimated.

**Parameters** (medians by stratum; `parameters.csv`): in the stratum medians the coupled estimator pools a little
and shrinks toward the identity less than LW — not in every class (β = 0, pooled only, in **no** class-fold; β = 1,
no pooling, in up to 3.4 % of a stratum; α = 1, no identity shrinkage, in 12–45 %; some classes shrink more than
under LW, e.g. *Sialis* sp. at 16 PCs 0.4883 vs 0.4383). The two identity targets have different scales (tr T_k/p vs
the biased class trace/p), so the coefficients alone do not show under-regularisation. Shrinkage toward identity
coupled vs LW, 64 PCs: FIN-Benthic2 0.1012 vs 0.2333 (15–63),
0.0050 vs 0.0392 (≥ 256); AquaMonitor 0.0582 vs 0.1536 (15–63), 0.0020 vs 0.0081 (≥ 256). β median 0.72–0.85 for
15–63 and 0.93–0.99 for ≥ 256. All covariances symmetric and positive definite; median condition numbers similar
(e.g. 64 PCs, ≥ 256: 259 vs 301 FIN-Benthic2, 418 vs 406 AquaMonitor, LW vs coupled).

**Corner checks** (labelled diagnostics, no tuning; `corners.csv`): pooled SCM alone beats LW in 6 of 350
FIN-Benthic2 and 1 of 408 AquaMonitor class-folds; the class's scaled identity in none; the class SCM (where n > d)
is worse on average everywhere. The coupled estimator is an interior combination, not one of its corners —
consistent with its specification.

## 4. Examples (64 PCs; `examples_by_fold.csv`)

| Taxon | Train specimens | Folds | LW shrink | Coupled shrink | β | Coupled − LW per fold (min, max; mean) |
|---|---|---|---|---|---|---|
| *Leuctra nigra* (FIN-Benthic2) | 15 | 3 | 0.487 | 0.258 | 0.457 | +3.012, +4.838; +3.848 (at 16 PCs −0.7863) |
| *Sialis* sp. (FIN-Benthic2) | 15.5 | 2 | 0.466 | 0.428 | 0.657 | +5.550, +15.589; +10.569 |
| *Mystacides* (AquaMonitor) | 37.5 | 4 | 0.224 | 0.038 | 0.707 | +1.331, +4.712; +3.121 |
| *Heptagenia dalecarlica* (AquaMonitor) | 137 | 5 | 0.062 | 0.000 | 0.899 | −4.635, −2.338; −3.104 |
| *Lepidostoma hirtum* (FIN-Benthic2) | 170 | 5 | 0.049 | 0.004 | 0.935 | −2.065, −0.873; −1.360 |
| Chironomidae (AquaMonitor) | 7,416 | 5 | 0.001 | 0.000 | 0.998 | −0.005, +0.009; +0.001 |

Medians over folds for shrinkage and β. The two mid-sized losers lose in all five folds while the coupled estimator
uses almost no identity shrinkage; this is compatible with section 8's finding that the squared-error-optimal
intensity is too small for held-out likelihood, but the run does not separate the identity-shrinkage, target-scale
and pooling effects.

## 5. Limits and negative findings

- **No general improvement:** no universal winner; no micro interval above 0. This compares two complete estimators
  (different SCM normalisation, identity target and tuning), not the isolated contribution of pooling.
- The rare-taxon gain rests on few taxa (7 FIN-Benthic2, 9 AquaMonitor; some eligible in only 2–4 folds) and on the
  64-PC setting; it is not seen at 16 PCs.
- Intervals reflect val-specimen sampling only (train fits fixed); folds overlap; pointwise, no multiplicity control.
- The coupled tuning targets Frobenius MSE under elliptical, finite-fourth-moment classes; our criterion is Gaussian
  likelihood and several classes are multimodal (life stages, acquisition source; earlier tasks), so its optimality
  does not transfer and was not assumed.
- The β = 0 vs Bhattacharyya point (task item 10): no class-fold has β = 0, and even then different α_k would give
  different final covariances; no statement about the log-determinant term is made.

## 6. Decisions for the author

1. Keep per-class LW as the main class model (a reasonable operational choice from this run, not a general
   superiority result), or not.
2. Whether to run the bounded **native-1024** comparison of §7 (separate approval; CPU only).
3. Whether to separate the mechanisms (pooling vs identity shrinkage vs target scale) — only if the central question
   needs it. A validation-derived hybrid rule ("coupled only where n_c ≤ d") is **not** recommended (the reviewer
   advises against it as well).

## 7. Methods review (Astra high) and the dimension question

**Asked** (author's request): validity of fidelity, conventions, paired design, bootstrap, strata and wording; and
whether native 1024-d frozen features are the more appropriate setting, since both estimators were proposed for
dimension comparable to or above the class size (Ledoit & Wolf 2004: p/n bounded, p may exceed n; Raninen & Ollila
2021 simulations: p = 200, n_k = 25–100) while we fitted in 16/64 train-PCA dimensions. Fresh read-only session,
gpt-6-astra high; handover `docs/evidence/coupled_rscm_vs_lw/astra_review/context_summary.md`; answer saved there
(`consult_stdout.txt`) and in `~/agent_consults/2026-11-12_162950_codex_gpt-6-astra_high.md`. Reviewed checkout e3d6e1e,
all file hashes matched.

**Verdict: needs changes** — implementation substantially faithful; several interpretations corrected. Each point was
checked against the code, saved outputs and the paper before acting:

| # | Finding | Checked | Action |
|---|---|---|---|
| 1 | Headline understated the primary (macro) outcome; "micro never better" wrong (two positive micro point estimates) | confirmed (`folds.csv`) | verdict rewritten |
| 2 | "n > d for every eligible class at 16 PCs" false (15–16 train specimens: rank ≤ n − 1 < 16); pooling mechanism untested | confirmed | §3 corrected |
| 3 | The paper itself clips γ̂_k to [1, p] (Sect. IV-B, [·]₁ᵖ) | confirmed on the rendered page | source note, wrapper docstring, §1 corrected |
| 4 | "Always pools" / "shrinks less than LW" hold for medians only; targets differ in scale | confirmed (`parameters.csv`, `examples.csv`) | §3, §4 qualified |
| 5 | reshape adaptation, equivalence check, pooling weights, Appendix A coefficients, conventions correct; 1/(n − 1) vs 1/n intentional | — | none |
| 6 | Pairing, strata, stratified bootstrap appropriate for the conditional question; intervals exclude training uncertainty | — | already in §5 |

**Dimension question — reviewer's answer:** native 1024-d would be **more informative** about the original frozen
representation and low-sample covariance estimation, but is not needed for the approved PCA study to be valid (LW
also covers fixed dimension, §3.1 Assumption 1; neither estimator requires p > n). What changes at p = 1024:
most class SCMs are singular; the Gaussian likelihood is weighted by the inverse covariance, so small estimated
eigenvalues and tail directions can dominate, while both tunings target Frobenius error; pooled train PCA keeps
high-total-variance directions (including between-class variation) and drops directions that may matter for
within-class density, especially of rare classes; more dimensions do not make the plug-ins reliable at n = 15
(sphericity result needs γ/p → 0; the paper reports bias for compound-symmetry spectra). **Either direction of change
is plausible.**

**Proposed bounded native run (not started; needs approval):** same caches, eligible classes, means, five folds and
both datasets; native coordinates through an explicit native-input path (not `--pcs 1024`, which still rotates, and
the authors' coordinatewise kurtosis is not exactly rotation-invariant); no whitening, no validation tuning;
CRSCM − LW compared within the dimension only (no cross-dimension comparison of gains); macro primary, micro
secondary, existing strata plus n_c/p and n_c ≤ p; per-fold/class differences with conditional intervals, parameters,
plug-in estimates and clip frequencies, minimum eigenvalues, conditioning, failures kept in the accounting (no jitter
or pseudo-inverse), runtime and memory; log-determinant and Mahalanobis parts reported separately; float64 Cholesky
scoring with finite/symmetry/pooled-rank checks; separate α and β convergence residuals monitored without changing
the estimator; one predetermined fold timed first, continuation decided on resources and numerics only. Cost: CPU
only; one class covariance at p = 1024 is cheap, and the authors' pairwise inner products need K(K − 1)/2
products of 1024 × 1024 matrices per fold (K = eligible classes) — an estimate would come from the timed fold.

Not done (per task): no replacement of LW in earlier tables, no GBC/Bhattacharyya recomputation, no FDA change, no
classifiers, no tuning on validation, no LINPOOL/ELL, no taxonomy pooling, no test sets.
