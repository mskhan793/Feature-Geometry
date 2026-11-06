# Research coding tasks — Feature-Geometry

## Current state
Repository scaffold initialized. No experimental claims or benchmarks have been run.

## Working rules
- Continue one mathematical idea at a time alongside our paper-study discussion.
- Do **not** implement all experiments at once. Each implementation task must be explicitly scoped and accepted independently.
- Prioritize numerical correctness, simple synthetic controls, then real frozen features.
- Keep all pretrained encoders frozen unless an experiment explicitly studies adaptation.
- Never use test labels to choose hyperparameters, projections, or shrinkage strengths.
- Record seeds, splits, feature normalization, sample counts, and exactly which covariance estimator is used.
- Distinguish population quantities from finite-sample estimates; note singular and ill-conditioned covariances.
- Preserve earlier findings and report negative results; no invented novelty.

## Task queue (planned, not started)
1. **NCM / metric geometry:** Compare Euclidean and cosine NCM with controlled class-mean separation and within-class variance. Later evaluate learned/shared Mahalanobis variants.
2. **H-score / shrinkage:** Implement original and shrinkage H-scores; test singularity, Ledoit–Wolf estimation, and common-dimensional random projection.
3. **Gaussian feature replay:** Sample embeddings from class-wise Gaussian estimates; check synthetic sample means/covariances, sensitivity to sample count, and classifier performance.
4. **Gaussian mixtures:** Fit multi-component Gaussian models to controlled multimodal feature distributions; compare prototype, single Gaussian, and GMM inference.

## Next small implementation task
**Experiment 03, first validation only:** Show that the empirical mean/covariance of synthetic Gaussian embeddings approach the specified generating mean/covariance with increasing sample count. Use a reproducible 2D toy example before any CLIP/BioCLIP data. This is a *suggested next task*, not an instruction to launch a large experiment.

## Definition of done for each task
- Minimal tested code in `src/` when appropriate
- Runnable script/notebook with deterministic seed
- Clear dataset/synthetic setup and evaluation metric
- Short results file with what worked, what failed, and limitations
- No hidden modifications to unrelated experiments
