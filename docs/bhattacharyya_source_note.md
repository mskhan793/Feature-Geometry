# Source note: Gaussian Bhattacharyya coefficient (Pándy et al.) — paper gate for task.md 0237ac7

## 1. Paper (downloaded and read)

| | |
|---|---|
| Paper | M. Pándy, A. Agostinelli, J. Uijlings, V. Ferrari, T. Mensink, *Transferability Estimation using Bhattacharyya Class Separability*, CVPR 2022 |
| Source | https://arxiv.org/abs/2111.12780 ; PDF https://arxiv.org/pdf/2111.12780v3 (latest = v3, 11 Apr 2022; v1 24 Nov 2021, v2 4 Apr 2022) |
| Local copy | `~/Documents/Research/methods/gbc_pandy2022/paper.pdf` (outside Git), 15 pages, 2,417,828 bytes |
| SHA256 | `18d703e361753507f7e2c4c0474e9969476bf0632c84558a607096d444c482e7` |

Page numbers below are the paper's printed page numbers.

| Item | Paper | Where |
|---|---|---|
| Bhattacharyya coefficient | BC(p_ci, p_cj) = ∫ √(p_ci(x) p_cj(x)) dx | Sect. 3.2, Eq. (1), p. 3 |
| Class Gaussian | p_c = N(μ_c, Σ_c); μ_c class mean of embeddings; Σ_c = 1/(N_c − 1) Σ_i (f(x_i) − μ_c)(f(x_i) − μ_c)ᵀ (unbiased) | Sect. 3.2, p. 3 |
| Bhattacharyya distance | D_B = ⅛ (μ_ci − μ_cj)ᵀ Σ⁻¹ (μ_ci − μ_cj) + ½ ln( \|Σ\| / √(\|Σ_ci\| \|Σ_cj\|) ), Σ = ½(Σ_ci + Σ_cj); BC = exp(−D_B) | Sect. 3.2, Eq. (2), p. 4 |
| Dataset-level GBC | GBC = − Σ_{i,j} [[i ≠ j]] BC(c_i, c_j): sum over **ordered** pairs (each unordered pair counted twice); higher BC = more overlap, so GBC is the negative sum | Sect. 3.2, Eq. (3), p. 4 |
| Gaussian assumption | "we do not suggest that per-class distributions are necessarily (multivariate) Gaussians" | Sect. 3.2, p. 4 |
| PCA | embeddings projected by PCA to 64 dimensions (to compare architectures with different feature sizes and to estimate the Gaussians better); whitening not stated; fitted on the target training set (Sect. 3.1 defines the target training set D_t) | Sect. 3.3, p. 4 |
| Covariance | full covariance "infeasible" for few samples per class; they "experiment with both diagonal covariance matrices and spherical ones" | Sect. 3.3, p. 4 |
| Chosen setting | spherical regularisation with 64-D PCA used "in all classification experiments" (CIFAR-10 study over {16,32,64,128} × {full, diagonal, spherical}) | Sect. 4.1 "Influence of regularization", p. 6; App. B, Figs. 6–7, p. 9 |
| Segmentation | diagonal covariance instead of spherical (many more observations) | Sect. 4.3, p. 8 |
| Spherical covariance | **not defined further** (which scalar variance, estimator, any floor) | — |

The pairwise coefficient BC(c_i, c_j) is a per-pair overlap of two fitted Gaussians; the dataset-level GBC score
(Eq. 3) is a single transferability score per model and dataset. The theoretical link to the Bayes error holds for a
fixed backbone and Gaussian classes (Sect. 3.2, p. 4).

## 2. Implementation

**The manuscript links no GBC code** (its only code links are for the baselines LEEP/LogME and H-score, footnote p. 5).

Candidate found: google-research `stable_transfer/transferability/gbc.py`, repository
https://github.com/google-research/google-research, commit `5b09c22d73a9d35eb6c5d2a99b95677a45053466` (latest commit
touching the file; copies with SHA256 in `~/Documents/Research/methods/gbc_pandy2022/code_candidate/`:
gbc.py `4f2dd55c…`). Attribution: the `stable_transfer` README states the code belongs to Agostinelli, Pándy, Uijlings,
Mensink, Ferrari, *How stable are Transferability Metrics evaluations?*, ECCV 2022 (arXiv:2204.01403) — the same five
authors; the file's docstring cites the GBC paper. It is therefore the authors' later implementation, **not linked from
the GBC manuscript**.

Further checks (2026-11-11): the `master` copy of gbc.py is byte-identical to commit 5b09c22; the first release
(commit `4963f325fe8bf41c69aeb2717d0ef408f623500e`, 2022-07-19) differs only in the copyright year. The ECCV 2022 paper
(arXiv:2204.01403, `~/Documents/Research/methods/gbc_pandy2022/eccv2022/paper.pdf`, SHA256 `78a67dfd…`) links
`github.com/google-research/google-research/tree/master/stable_transfer` as "the source code for our experiments"
(footnote 2) and describes GBC only in words (Sect. 3.1); it does not define the spherical variance or say this code
produced the CVPR 2022 GBC results. Label: **the authors' published GBC implementation (ECCV 2022 code release), not
verified as the code behind the GBC paper's own results.**

What the code does (read, not run; TensorFlow):
- per class: mean and **population** variance per dimension (`tf.math.reduce_variance`, divides by N), floored at
  1e-4;
- `spherical` (the default in its source-selection config, with `pca_reduction = 64`): each class's variance is the
  **mean of its per-dimension variances**, a scalar s_c;
- distance: ⅛ Σ_k (μ_ik − μ_jk)² / s̄ + ½ [ln s̄ − ½ ln s_i − ½ ln s_j], with s̄ = (s_i + s_j)/2;
- PCA: `sklearn.decomposition.PCA(n_components=64).fit_transform` on the target training features (not whitened);
- GBC: −Σ over ordered pairs c1 ≠ c2 of exp(−D_B) (same as Eq. 3).

**Differences from the manuscript (material):**
1. Spherical log-determinant term. For Σ_c = s_c I_d, Eq. (2) gives ½ ln(\|Σ\| / √(\|Σ_i\|\|Σ_j\|)) =
   **d** · ½ [ln s̄ − ½ ln s_i − ½ ln s_j]. The code's term has no factor d (= 64), so its covariance-scale part is
   1/64 of Eq. (2)'s for spherical Gaussians. In `diagonal` mode the code sums over dimensions and matches Eq. (2) for
   diagonal Σ.
2. Variance estimator: population (÷N) in the code vs unbiased (÷(N_c − 1)) in the paper's Σ_c.
3. Variance floor 1e-4: in the code only.
4. Spherical scalar = mean per-dimension variance: in the code only (the paper does not say).

Consequence for this task: with a spherical covariance there is **no covariance orientation** at all (σ²I is
isotropic). The task's orientation question needs a non-spherical covariance (the paper's App. B variants: full or
diagonal; only full covariance has orientation).
