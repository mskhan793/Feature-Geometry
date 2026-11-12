# Context for the methods review — coupled RSCM vs per-class Ledoit–Wolf (2026-11-12)

## What we are doing now (task.md 61ceb61, active section "Focused covariance study", lines 72–109)

Question: does estimating each taxon's covariance by borrowing from other taxa (coupled regularized SCM, CRSCM)
improve **held-out class-conditional Gaussian log-likelihood** over our per-class Ledoit–Wolf (LW) full covariance,
especially for rare taxa? Frozen DINOv3-L specimen embeddings (1024-d; specimen = mean of camera means of up to 10
frames per camera), FIN-Benthic2 (derived five-split CSV) and AquaMonitor Classification (official five folds); train
fits, val scores, no test. **Fixed by the task: train-fitted PCA, 16 and 64 dimensions.** Code commit `8cf3738`
(results `e3d6e1e`). Results and verdict: `response.md`; tables: `docs/evidence/coupled_rscm_vs_lw/summary/summary.md`;
method/provenance: `docs/coupled_rscm_source_note.md`.

Summary of results (coupled − LW): no overall gain (specimen-micro interval below 0 in 15 of 20 fold × dimension runs,
above 0 in none); rare taxa (15–63 train specimens) gain at 64 PCs (+2.97 FIN-Benthic2, +1.10 AquaMonitor; interval
> 0 in 4/5 folds) but lose slightly at 16 PCs; classes ≥ 64 lose slightly; coupled shrinks toward identity much less
than LW; β = 0 never.

## What we tried before (same repository, all on the same embeddings)

1. Five-fold class geometry and Gaussian mixtures in 8/16 train-PCA (whitened) dims; Astra high methods review
   (record `~/agent_consults/2026-11-11_183354_codex_gpt-6-astra_high.md`), corrections `e0f39a5`.
2. Bhattacharyya (Pándy et al. 2022, Eq. 2) vs d′, fold 0, 8/16/32 PCs (`abb8ac9`): spherical vs full covariance.
3. PCA-GBC vs FDA-GBC (Mensink et al. 2013, Eq. 8 scatter, LW-shrunk S_W), five folds, PCA from native 1024-d;
   FDA in native 1024-d (`cb5c21a`, `87f0542`).
4. Ledoit–Wolf diagnostics (`831c318`, `00e6979`; `docs/evidence/lw_fit/summary/summary.md`): pooled within-class
   S_W in **native 1024-d** — α_LW (0.007 FIN-Benthic2, 0.001 AquaMonitor) under-shrinks for held-out likelihood
   (fixed α = 0.01 better in all 10 runs); frames/specimens per class. Per-class covariances only in PCA 16/64:
   per-class LW best of LW / sample / diagonal / spherical at almost every class size; spherical worst.
5. Literature (response.md §9 at `ea56c64`) → this task.

## References (local copies)

- Raninen & Ollila, Coupled Regularized SCM Estimator for Multiple Classes, IEEE TSP 69:5681–5692, 2021 (text:
  `~/Documents/Research/methods/Shrinkage/Aalto_thesis/Coupled_Regularized_Sample_Covariance_Matrix_Estimator_for_Multiple_Classes.txt`);
  simulations p = 200, n_k = 25–100 (also 10–200), K = 4, multivariate t; real data RDA (Sonar, Vowel, Ionosphere).
- Ledoit & Wolf, A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices, JMVA 2004 (preprint text
  `~/Documents/Research/methods/ledoit_wolf2004/paper.txt`): asymptotics with p/n bounded, p may exceed n.
- Also in the folder (not used here): Raninen, Tyler & Ollila, Linear Pooling of SCMs, IEEE TSP 2022; Ollila &
  Raninen, Optimal Shrinkage under Elliptical Distributions, IEEE TSP 2019; Raninen 2022 thesis.

## The author's new question for this review

Both estimators were proposed for the high-dimensional regime (p comparable to or larger than n_k). We fitted them in
16/64 train-PCA dimensions, where most classes have n_k > p. **Would the native 1024-d frozen feature space be the
more appropriate (or more informative) setting for this comparison?** Native 1024-d: n_k/p from about 0.015 (15
specimens) to 7 (Chironomidae 7,416); the class SCM is singular for most classes there.
