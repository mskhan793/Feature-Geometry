# Method note: PCA-GBC vs regularised FDA-GBC (task.md eeb9707)

## 1. Sources (local copies read in full)

| | Local copy | SHA256 | Code |
|---|---|---|---|
| Pándy et al., *Transferability Estimation using Bhattacharyya Class Separability*, CVPR 2022, arXiv:2111.12780v3 | `~/Documents/Research/methods/gbc_pandy2022/paper.pdf` | `18d703e3…` | none linked; authors' later release audited in `docs/bhattacharyya_source_note.md` |
| Mensink, Verbeek, Perronnin, Csurka, *Distance-Based Image Classification: Generalizing to New Classes at Near-Zero Cost*, IEEE TPAMI 35(11):2624–2637, 2013 (UvA accepted manuscript, https://pure.uva.nl/ws/files/19634478/MensinkTPAMI2013.pdf, DOI 10.1109/TPAMI.2013.83) | `~/Documents/Research/methods/ncm_mensink2013/paper.pdf` (15 pages; identical copy in `Feature_geometry_data/Aqua_Paraquete_files/Papers/`) | `5dfaa5a34989971e29f4529fe917864c51e1c1c841462d410f0bffdcff839aca` | none linked (only DOI and repository URLs in the text) |

Mensink et al. give FDA in Sect. 3.2, Eq. (8): find W maximising L_FDA = tr( (W S_W Wᵀ)⁻¹ (W S_B Wᵀ) ), with
S_B = Σ_c (N_c/N)(μ − μ_c)(μ − μ_c)ᵀ (weighted covariance of the class means, μ the data mean) and
S_W = Σ_c (N_c/N) Σ_c (weighted sum of within-class covariances), referring to Webb (2002, textbook, not available
locally) for details. Their Fig. 1 illustrates that FDA can collapse nearby classes (FDA maximises total
between-class spread, not the separation of the closest classes). Neither paper gives code for FDA or for the
estimator of Σ_c.

## 2. Our choices (not from the papers)

- **Unit and labels:** one embedding per independent training specimen (mean of camera means), N_c = training
  specimens of class c, all classes present in the fold's training split (also those below the GBC support rule).
- **Σ_c in S_W:** biased (÷N_c), so S_W = (1/N) Σ_i (x_i − μ_{y_i})(x_i − μ_{y_i})ᵀ, the pooled within-class scatter
  over N; with Mensink's weights this makes S_W and S_B the within- and between-class parts of the biased total
  covariance (S_T = S_W + S_B). The class weights N_c/N follow Eq. (8); in AquaMonitor large classes dominate both.
- **Regulariser R (shrinkage, not ridge):** S_W is 1024 × 1024 and estimated from a few thousand specimens, so its
  smallest eigenvalues are underestimated and the generalised problem is ill-conditioned. We use Ledoit–Wolf
  shrinkage toward a scaled identity: S_W* = (1 − α) S_W + α (tr S_W / D) I, i.e. R = α ((tr S_W / D) I − S_W), with
  the intensity α ∈ [0, 1] estimated analytically from the training residuals x_i − μ_{y_i}
  (`sklearn.covariance.ledoit_wolf_shrinkage`, residuals already centred). Units: those of the features (variance).
  No grid search; α/2 and 2α (capped at 1) only as labelled sensitivity. S_W* is checked positive definite by
  Cholesky and its condition number is reported.
- **Solver:** symmetric-definite generalised eigenproblem S_B v = λ S_W* v by `scipy.linalg.eigh(S_B, S_W*)` (no
  explicit inverse). rank(S_B) ≤ C_train − 1, so at most C_train − 1 eigenvalues are non-zero (38 for 39 classes,
  41 for 42); 64 FDA dimensions are impossible and are never padded. Dimensions d ∈ {8, 16, 32} are used only if
  d ≤ C_train − 1 and the d-th eigenvalue is numerically non-zero (> 1e-10 × the largest).
- **Projection-scale convention (main):** the top-d generalised eigenvectors are orthonormalised (QR); features are
  centred on the training mean and projected onto this orthonormal basis, exactly like PCA (also centred, orthonormal,
  unwhitened). PCA-d and FDA-d then differ only in *which* d-dimensional subspace they keep, not in scaling.
  Sensitivity: raw generalised eigenvectors scaled to vᵀ S_W* v = 1 (FDA's own whitened metric), labelled.
- **Comparisons:** PCA-d vs FDA-d at matched d ∈ {8, 16, 32}; PCA-64 spherical is the paper's baseline, reported
  separately (no FDA-64). Gaussian Bhattacharyya by Pándy Eq. (2) (src.methods.bhattacharyya), spherical main,
  diagonal and full as sensitivity (full only with ≥ 2 d specimens per class).

## 3. Equal means, different covariances (task part A)

**Bhattacharyya distance** (Pándy Eq. 2), Σ̄ = ½(Σ_a + Σ_b):
D_B = ⅛ Δμᵀ Σ̄⁻¹ Δμ  +  ½ ln( |Σ̄| / √(|Σ_a| |Σ_b|) ) = mean term + covariance term.

- **Shared covariance** (Σ_a = Σ_b = Σ): covariance term 0, D_B = ⅛ Δμᵀ Σ⁻¹ Δμ (⅛ of the squared Mahalanobis
  distance).
- **Equal means** (Δμ = 0): mean term 0, D_B = ½ ln( |Σ̄| / √(|Σ_a||Σ_b|) ) ≥ 0, with equality only if Σ_a = Σ_b
  (by the concavity of ln det). So D_B > 0 (BC < 1) whenever the covariances differ, even with identical means.
- **Eigenvalues and orientation.** If Σ_a, Σ_b share eigenvectors with eigenvalues λ_k, κ_k:
  covariance term = ½ Σ_k ln( (λ_k + κ_k) / (2 √(λ_k κ_k)) ) = ½ Σ_k ln( (r_k + 1/r_k)/2 ), r_k = √(λ_k/κ_k),
  each summand ≥ 0 and 0 only if λ_k = κ_k. With the **same eigenvalues but rotated eigenvectors**, |Σ_a| = |Σ_b|
  while |Σ̄| is larger (averaging differently oriented ellipsoids makes a rounder, larger-volume one), so the
  covariance term is positive from orientation alone.

**Hand-checked 2-D examples (equal means):**
1. Orientation only: Σ_a = diag(4, 1), Σ_b = diag(1, 4) (same eigenvalues, axes swapped). Σ̄ = 2.5 I, |Σ̄| = 6.25,
   |Σ_a| = |Σ_b| = 4: D_B = ½ ln(6.25 / 4) = ½ ln 1.5625 = 0.2231, BC = 1.5625^(−½) = **0.8**.
2. Scale only: Σ_a = I, Σ_b = 4 I. Σ̄ = 2.5 I, |Σ̄| = 6.25, √(|Σ_a||Σ_b|) = √16 = 4: the same D_B = 0.2231, BC = 0.8.
Both are verified in `tests/test_pca_fda_gbc.py`.

**What d′ can and cannot see.** Our d′ uses u = (μ_a − μ_b)/‖μ_a − μ_b‖ and
d′ = |uᵀ(μ_a − μ_b)| / √(½(uᵀΣ_a u + uᵀΣ_b u)) = |uᵀΔμ| / √(uᵀ Σ̄ u).
- With equal means u is **undefined** (0/0), not zero: d′ is not a measurement of covariance difference and must be
  reported as undefined, never as 0. With nearly equal means u is dominated by sampling noise.
- d′ uses the covariances only through the single number uᵀΣ̄u, the average variance **along u** (its denominator).
  Two pairs with the same means and the same variance along u have the same d′ however different their
  ellipsoids are in the other directions; D_B's covariance term sees those differences.
- **Link between the two (Cauchy–Schwarz):** for the same Σ̄, ⅛ Δμᵀ Σ̄⁻¹ Δμ ≥ ⅛ (uᵀΔμ)² / (uᵀΣ̄u) = d′²/8, with
  equality iff Σ̄⁻¹Δμ ∥ Δμ (u is an eigenvector of Σ̄). So the Bhattacharyya mean term is at least d′²/8 and exceeds
  it when the average covariance is tilted relative to the mean difference (the Mahalanobis direction Σ̄⁻¹Δμ finds a
  better separating direction than u). This holds exactly for the full-covariance mean term and the train d′ computed
  on the same specimens with the same (n − 1) estimator; it is checked numerically in the tests and on real data
  (`summary.json`, `mean_term_ge_dprime2_over_8`).
