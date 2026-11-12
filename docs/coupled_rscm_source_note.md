# Coupled RSCM vs per-class Ledoit–Wolf — source note (task.md 61ceb61)

## Primary sources (opened and read locally)

| Source | Version / URL | Local path | SHA256 |
|---|---|---|---|
| Raninen & Ollila, *Coupled Regularized Sample Covariance Matrix Estimator for Multiple Classes*, IEEE TSP 69:5681–5692, 2021 | published version (CC BY 4.0), doi [10.1109/TSP.2021.3118546](https://doi.org/10.1109/TSP.2021.3118546); 12 pages | `~/Documents/Research/methods/Shrinkage/Aalto_thesis/Coupled_Regularized_Sample_Covariance_Matrix_Estimator_for_Multiple_Classes.pdf` (supplied by the author, 2026-11-12) | `14853bade1f89a1d3a4e0645569871bc3cb2ef7de3e5b50c6c6f9c6aab0f97cf` |
| Ledoit & Wolf, *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*, JMVA 2004 | authors' preprint Feb 2001, http://www.ledoit.net/ole1a.pdf | `~/Documents/Research/methods/ledoit_wolf2004/paper.pdf` | `08e79244bf0ea47acb4a35189cfd7614bbc21cc058684da3d0cfe50d83717f1c` |
| Shrinkage literature folder index (thesis + 6 papers, code archives) | — | `~/Documents/Research/methods/Shrinkage/README.md` | `64008512ee617f6c77ab5cfb952464d280d022368ea1789fc8ef8f20828dec4b` |

## Authors' code

- Repository https://github.com/EliasRaninen/CoupledRSCM, commit `bb09e1230b5d9b0e03dbb7a5d9850ee9e18399a3` (archive sha256
  `84d001474fdeb4b9ce3481a0d06c7701d07cc0c517cbb23caed7484813a5f079`), MIT licence. File used:
  `Python/multiclass_rscm.py` ("This version: March 2022"), sha256
  `a67e443d18613b2671f6997059cf900a3a51853d5584f7251e6a290b780aa326`, vendored byte-identical in
  `third_party/coupled_rscm/` with its `LICENSE` (identity checked by `tests/test_coupled_rscm.py`).
- Dependencies: numpy, pandas, `scipy.stats.stats.kurtosis` (deprecated namespace; works with SciPy 1.18.0, the version
  in the LUMI container and in `.venv-gpu`, with a DeprecationWarning; removed in SciPy 2.0), numpy 2.3.5, pandas 2.3.3.
- Functions used: `estimate_parameters(dataset)` (pandas DataFrame, features + a `class` column coded 0..K−1) →
  plug-in parameters and per-class SCMs; `rscmpool(params, averaging=False, compute_inverse=False)` → `Sigmas`,
  `alpha`, `beta` (one value per class). Not used: `rscmpools` (streamlined variant), `linpool` (LINPOOL, out of scope),
  `compute_inverse_rscm` (only faster when p > N).

## Estimator (paper Eq. 1, 2, 4; code `rscmpool`)

S_k = (1/(n_k − 1)) Σ_i (x_ik − x̄_k)(x_ik − x̄_k)ᵀ (unbiased), S = Σ_k π_k S_k with π_k = n_k / Σ_j n_j,

T_k(β_k) = β_k S_k + (1 − β_k) S,   Σ̂_k = α_k T_k(β_k) + (1 − α_k) (tr T_k(β_k) / p) I.

- **Class-specific:** every class has its own (α_k, β_k) (code `averaging=False`; `averaging=True` would be the
  paper's POLY-Ave, not used). The pooled matrix S is shared and covers the classes passed in.
- **Parameter meaning:** α_k is the weight **kept** on the data matrix T_k; the shrinkage toward the scaled identity is
  1 − α_k. Our / scikit-learn's Ledoit–Wolf α is the shrinkage itself. Reported side by side as
  `coupled_shrink_to_identity` = 1 − α_k and `lw_alpha_shrink`; they are never compared as the same quantity.
  β_k is the weight on the class's own S_k (β_k = 0: the pooled S only).
- **Tuning target:** (α_k, β_k) minimise a plug-in estimate of E‖Σ̂_k − Σ_k‖²_F, a biconvex polynomial (Theorem 1).
  Search (Sect. V-A, code): grid {0, 0.05, …, 1}², then alternating closed-form updates (7)/(8) clipped to [0, 1]
  until the relative change < 1e-8 (at most 1000 iterations).
- **Plug-in parameters** (Sect. IV, Lemma 1, code `estimate_parameters`): η_k = tr(S_k)/p; κ_k = max(mean_j k̂_j / 3,
  −2/(p + 2)) with k̂_j = m4/m2² − 3 (biased moments; `scipy.stats.kurtosis` default; coordinatewise, so not
  exactly rotation-invariant); γ_k = [(p n_k/(n_k − 1))(‖S̃_k‖²_F − 1/n_k)]₁ᵖ (clipped to [1, p]) with S̃_k the spatial sign covariance around the spatial median; ⟨Σ_i, Σ_j⟩_F = p² η_i η_j ⟨S̃_i, S̃_j⟩_F (i ≠ j),
  ‖Σ_k‖²_F = p γ_k η_k²; E‖S_k‖²_F and E[tr(S_k)²] from Lemma 1 with τ1 = 1/(n_k − 1) + κ_k/n_k, τ2 = κ_k/n_k.
- **Assumptions:** classes are independent samples from (unspecified) elliptical distributions with finite fourth
  moments; the optimality is for Frobenius MSE, not Gaussian likelihood.

## Paper vs code (checked line by line)

| Item | Paper | Code | Agreement |
|---|---|---|---|
| Eq. (4) estimator | as above | `alpha[k]*Sb + (1-alpha[k])*trace(Sb)/p*I`, `Sb = beta*SCM_k + (1-beta)*Sp` | equal |
| SCM, pooled SCM | unbiased, π_k = n_k/N | `scov` (n − 1), `Sp = Σ n_k SCM_k / N` | equal |
| MSE coefficients C_ij (Appendix A) | C̃22 … C̃00 | `C["22"]` … `C["00"]` | equal (each term checked) |
| κ̂_k, η̂_k, γ̂_k, inner products | Sect. IV | `estimate_kappa`, `estimate_eta`, `estimate_gamma`, `trCiCj` | equal |
| γ̂_k clip | estimate clipped, γ̂_k = [·]₁ᵖ (Sect. IV-B; clamp notation defined in Sect. I) | `np.clip(gam, 1, p)` | equal (corrected 2026-11-12 after the methods review: an earlier version of this note wrongly said the paper only states the range) |
| Optimisation | grid 0.05 + alternating (7)/(8) | same | equal; the code's slow-convergence warning is unreachable (`iteration == iterMAX` never true), and its stopping rule ‖Δα + Δβ‖/‖α₀ + β₀‖ adds the two changes before the norm, so they can cancel; convergence is not reported |
| Number of classes | any K | grid evaluation `msefunction(...).reshape(4)` | **code works only for K = 4** |

## Our adaptations (distinct from the original code and from the paper)

1. **K = 4 hard-code** (essential): in `rscmpool`, `C).reshape(4)` → `C).reshape(K)`, applied in memory when
   `src/methods/coupled_rscm.py` loads the source (asserted to occur exactly once; the file is not edited). Equivalence:
   on 4-class fixtures and, in every LUMI run, on the first four eligible classes of each real fold and PCA dimension,
   the unmodified and adapted code give identical α, β and Σ̂ (max abs difference required to be exactly 0, else the
   run fails). The original raises `ValueError` for K ≠ 4 (test).
2. **Convergence audit, output only** (added 2026-11-12 after the methods review): after the alternating search,
   `rscmpool` also stores its MSE-polynomial coefficients and iteration count in `params` (`_rscmpool_C`,
   `_rscmpool_iterations`; in-memory, asserted to occur once). `convergence_residuals` applies the authors' clipped
   updates (7)/(8) once more to the returned (α, β) and reports |Δα| and |Δβ| separately. Nothing in the estimate
   changes (the 4-class equality with the unmodified code still holds, test).
3. **Input format:** our PCA-projected train specimen embeddings (float64) for the eligible classes, coded 0..K−1 in
   sorted class order, passed as the DataFrame the code expects. No change to parameters or assumptions.
4. **Class means for the likelihood:** the train sample mean of each class (as for the LW reference); the authors' code
   estimates covariances only.

## Reference estimator (unchanged convention, section 8 D3)

Per-class Ledoit–Wolf: biased class covariance (1/n_c) shrunk toward (tr/d) I with the intensity α_LW of the class
residuals (`src/statistics/covariance_fit.lw_class_covariance`; the formula previously inline in
`scripts/lw_fit_diagnostics.py`, now shared by both scripts). Note the different normalisation (1/n vs 1/(n − 1)) of
the two estimators' data matrices: each is used as defined by its own source.
