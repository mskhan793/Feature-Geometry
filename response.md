# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 0237ac7, "Small addition: Bhattacharyya class overlap versus existing d′".
**Status: paper gate passed for the manuscript; STOPPED before implementation because the method is materially
underspecified for the paper's chosen setting and the authors' code differs from the manuscript (task item 4). Three
questions below.** Nothing implemented or run; no features read.

Previous response (corrected five-fold geometry, task 92a605a): commit c05a7e6.

## 1. Paper and code provenance

Details with equations and page numbers: `docs/bhattacharyya_source_note.md`.

- **Paper downloaded and read:** arXiv:2111.12780v3 (11 Apr 2022, latest of 3 versions), 15 pages,
  `~/Documents/Research/methods/gbc_pandy2022/paper.pdf`, SHA256 `18d703e361753507f7e2c4c0474e9969476bf0632c84558a607096d444c482e7`.
- **Formulas (paper):** class Gaussian N(μ_c, Σ_c) with unbiased Σ_c (Sect. 3.2, p. 3); Bhattacharyya distance
  D_B = ⅛ Δμᵀ Σ̄⁻¹ Δμ + ½ ln(|Σ̄| / √(|Σ_i||Σ_j|)), Σ̄ = ½(Σ_i + Σ_j), BC = exp(−D_B) (Eq. 2, p. 4); dataset-level
  GBC = −Σ over ordered pairs i ≠ j of BC (Eq. 3, p. 4). Pairwise BC (overlap of two fitted Gaussians; near 1 = more
  overlap) is distinct from the dataset-level GBC score.
- **Setting (paper):** 64-D PCA (Sect. 3.3, p. 4) and spherical covariance, chosen on CIFAR-10 over {16, 32, 64, 128}
  × {full, diagonal, spherical} (Sect. 4.1, p. 6; App. B, p. 9). The paper does **not** define the spherical
  covariance (which scalar variance, which estimator) and does not say whether the PCA is whitened.
- **Code:** the manuscript links no GBC code. An implementation exists in google-research `stable_transfer`
  (`transferability/gbc.py`, commit `5b09c22d73a9d35eb6c5d2a99b95677a45053466`), belonging to the same five authors'
  ECCV 2022 paper *How stable are Transferability Metrics evaluations?* and citing the GBC paper. Copies with SHA256 in
  `~/Documents/Research/methods/gbc_pandy2022/code_candidate/`.
- **Code vs manuscript (spherical setting, the paper's default):**
  1. its covariance term lacks the factor d: for spherical Gaussians Eq. (2) gives d · ½[ln s̄ − ½ ln s_i − ½ ln s_j],
     the code ½[ln s̄ − ½ ln s_i − ½ ln s_j] (1/64 of Eq. 2 at 64 PCs);
  2. population variance (÷N) instead of the paper's unbiased Σ_c (÷(N_c − 1));
  3. variance floor 1e-4 (code only);
  4. spherical scalar = mean of per-dimension variances (code only; the paper is silent);
  PCA: unwhitened sklearn PCA to 64 on the target training features. Its diagonal mode matches Eq. (2) for diagonal Σ.
- **Consequence for the task's question:** a spherical covariance (σ²I) has no orientation, so the paper's own setting
  cannot show covariance-orientation effects; only a full covariance can (the paper's App. B "full" variant).

## 2. Questions before implementation

- **Q1 — main pairwise BC formula (64 PCs, spherical):**
  (a) the authors' `stable_transfer` code exactly (no factor d, population variance, floor 1e-4) — reproduces their
  executable GBC; (b) Eq. (2) exactly for spherical Σ_c = s_c I (factor d, unbiased variance, s_c = mean per-dimension
  variance, no floor) — the manuscript's mathematics, with s_c defined by us; (c) both, (b) as the main pairwise
  Gaussian Bhattacharyya and (a) as the reproduction of the published code. *Recommendation: (c).*
- **Q2 — orientation:** add the paper's full-covariance variant (Eq. 2 with unbiased Σ_c, no extra regularisation,
  only for classes with enough independent train specimens, e.g. n ≥ 2 × 64, and the rest reported as ineligible), and
  the diagonal variant, as clearly labelled sensitivity analyses at 64 PCs? Without the full variant the orientation
  part of the task cannot be answered. *Recommendation: yes; plus 8 / 16 PCs only as labelled sensitivity.*
- **Q3 — which specimens estimate the Gaussians:** the paper fits class Gaussians on the target training set. For the
  existing d′, the direction comes from train and the spread from val specimens. Proposal: BC from **fold-0 train**
  specimen Gaussians (paper-like), and the same BC re-estimated from **fold-0 val** specimens in the same train-fitted
  PCA as a held-out check; d′ recomputed in the same 64-PC space (train direction, val spread) next to the native
  1024-D d′. *Recommendation: yes.*

PCA: unwhitened (as in the code; the paper is silent), fitted on fold-0 train specimen embeddings only — say if you
want otherwise.

## 3. Not done

No code, tests, LUMI jobs or feature access for this task yet. Earlier results and evidence unchanged.
