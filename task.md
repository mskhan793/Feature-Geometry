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


## Mandatory implementation architecture (overrides prior generic scaffold suggestions)

Use the same **PyTorch Lightning design pattern** as the user's existing `mskhan793/Lightning_Aqua_Vision_Language_v1` repository. Its `src/aquamonitor_clip/dataset.py` defines `HoyeDataModule`, `FinBenthic2DataModule`, and `AquaMonitorDataModule` as `pl.LightningDataModule`; `src/aquamonitor_clip/model.py` provides a `pl.LightningModule`. These are architectural references only: reuse the clear class/lifecycle pattern, **not** the old project's dataset-specific logic.

1. **Every runnable experiment uses a real Lightning data pipeline:** appropriately scoped `torch.utils.data.Dataset` classes and `pl.LightningDataModule` with distinct train/validation/test dataloaders; add `predict_dataloader` when needed.
2. **Any trained adapter, probe, or other learnable model** must be a `pl.LightningModule` and run through `pl.Trainer`. Keep training, validation, test hooks and optimizer configuration explicit.
3. **Training-free methods remain training-free.** Implement NCM/H-score/covariance/GMM estimators in separate, small `src/methods/<method>.py` files (or `src/statistics/` for shared estimators). Use Lightning for data loading/evaluation; do not fabricate optimization steps just to satisfy the framework.
4. **One method = one clear module**, and shared code is centralized. No dead/unused code, oversized multi-purpose scripts, copied implementations, hidden preprocessing, or empty scaffolds that serve no task.
5. **Split discipline:** fitting on train only, tuning/model selection on validation, final measurement on held-out test. Synthetic experiments likewise use seeded nonoverlapping data. Preserve specimen/observation-level grouping when relevant.
6. **Reproducibility:** explicit configs, seeds, dataset versions, normalization, estimator regularization, saved predictions and metrics, checkpointing only where training applies, unit and data-module integration tests.
7. **Scope discipline:** this chat discusses mathematical theory and interprets results; separate coding sessions act on the single explicit task written below. Before implementation, discuss and approve that task's scope; never silently implement the entire queue.

## Current approved task

**Architecture documentation only.** This update records the Lightning standard. No training workflows or new experiments are approved or claimed complete.

## Proposed next task (needs discussion first)

Refactor/extend the initial Gaussian replay toy into an end-to-end but minimal **Lightning experiment** with seeded 2D synthetic train/validation/test splits, a `LightningDataModule`, Gaussian sampling as its own method file, a meaningful evaluation path (and a `LightningModule` only if training a classifier). Check synthetic mean/covariance convergence; do not train unnecessary networks.


## Task queue (planned, not started)
1. **NCM / metric geometry:** Compare Euclidean and cosine NCM with controlled class-mean separation and within-class variance. Later evaluate learned/shared Mahalanobis variants.
2. **H-score / shrinkage:** Implement original and shrinkage H-scores; test singularity, Ledoit–Wolf estimation, and common-dimensional random projection.
3. **Gaussian feature replay:** Sample embeddings from class-wise Gaussian estimates; check synthetic sample means/covariances, sensitivity to sample count, and classifier performance.
4. **Gaussian mixtures:** Fit multi-component Gaussian models to controlled multimodal feature distributions; compare prototype, single Gaussian, and GMM inference.

## Definition of done for each task
- Minimal tested code in `src/` when appropriate
- Runnable script/notebook with deterministic seed
- Clear dataset/synthetic setup and evaluation metric
- Short results file with what worked, what failed, and limitations
- No hidden modifications to unrelated experiments
