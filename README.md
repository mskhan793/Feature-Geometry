# Feature-Geometry

A research laboratory for understanding **frozen pretrained visual representations** through geometry, statistics, and controlled classification experiments.

## Research questions
- When are class means enough for classification, and when does class covariance help?
- How do sample size, dimensionality, and covariance shrinkage affect statistical reliability?
- Can transferability scores predict practical frozen-feature classifier performance?
- When does a class require multiple prototypes or Gaussian-mixture components?
- How do these findings change under few-shot, class imbalance, taxonomy shifts, and OOD?

We study published methods **before** claiming new contributions. Synthetic-data tests verify mathematical behavior; conclusions about real visual representations require real, independently evaluated frozen embeddings.

## Repository structure
```text
experiments/
  01_ncm_metric_learning/    # NCM, distance metrics, class prototypes
  02_hscore_shrinkage/       # H-score, covariance stability, random projections
  03_gaussian_feature_replay/# Gaussian synthetic feature sampling
  04_gaussian_mixtures/      # Multimodal classes and mixture-based inference
src/
  lightning_utils/           # Shared Lightning pieces: seeding, run folder, callbacks, Trainer defaults
  statistics/                # Covariance, mean, shrinkage and related estimators
  classifiers/               # Prototype and statistical classifiers
  evaluation/                # Accuracy, balanced metrics, validation utilities
notebooks/                   # Educational exploratory analyses
tests/                       # Deterministic correctness tests
results/                     # Local outputs, never large committed artifacts
task.md                      # Current coding-agent task and research workflow
```


## Architecture contract: PyTorch Lightning

**This project uses PyTorch Lightning for all runnable training and evaluation workflows**, following the organizational pattern of the existing `Lightning_Aqua_Vision_Language_v1` repository (not a blind copy of its domain-specific implementation).

- Dataset classes: subclasses of `torch.utils.data.Dataset`, each with one clear responsibility.
- Data modules: subclasses of `pytorch_lightning.LightningDataModule`, with `setup(stage)`, `train_dataloader()`, `val_dataloader()`, and `test_dataloader()`. Use `predict_dataloader()` where appropriate. Respect independent train/val/test splits; for synthetic experiments, use seeded, nonoverlapping splits.
- Learnable models: subclasses of `pytorch_lightning.LightningModule`, with `training_step`, `validation_step`, `test_step`, and `configure_optimizers`. Only implement steps that are actually needed.
- **Training-free methods** (NCM, H-score, frozen statistical classifiers) must not invent optimizer training: keep each algorithm in its own modular Python file and use Lightning DataModules/Trainer evaluation or a thin Lightning evaluation wrapper when useful.
- Use `pl.Trainer` for real training, validation and testing, Lightning checkpointing and logging where relevant. Decide model selection using validation only, then evaluate held-out test once.
- One method per focused file. No long monolithic scripts, unused classes, duplicate implementations, unrelated branches, or dead code. Common math/data/metric helpers belong in shared modules with tests.
- Keep raw data access, feature extraction, statistical fitting, decision rules, and metrics separate; method implementations must not hide dataset-specific paths, splits, or research decisions.
- Prefer reproducible configurations and CLI entry points, explicit RNG seeds, output directories and experiment metadata.
- Existing `src/statistics/gaussian.py` is a small pure numerical utility, **not** a finished Lightning experiment. Build runnable Lightning components incrementally as their tasks are agreed.

### Planned architecture

```text
src/
  data/
    datasets/             # each dataset in its own Python file
    datamodules/          # LightningDataModule implementations
  methods/                # one statistical method/classifier per Python file
  lightning_modules/      # LightningModule training/evaluation orchestration
  statistics/             # shared estimators and numerical utilities
  classifiers/            # shared prediction interfaces/utilities as needed
  evaluation/             # split-aware metrics and reporting
  config/                 # explicit experiment configurations
experiments/              # small, independently runnable study definitions
tests/                    # unit and integration tests for data, methods, trainers
```

Create new subdirectories **only when their first real implementation is added**; avoid committing unused placeholders. The above paths describe the intended target layout, not files that already exist.

## Separation of responsibilities

This conversation is for mathematical derivations, interpretation of papers and experimental results. Coding agents/sessions use `task.md` as the source of truth for the **currently approved** implementation task. We discuss the specific experiment first; only then do we edit its task specification and implement it. Do not silently launch the entire backlog.

## Setup
Use Python >=3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Research workflow
1. Write a focused hypothesis and a minimal reproducible experiment in the relevant `experiments/` folder.
2. Derive expected behavior on a simple synthetic dataset first.
3. Implement and test the method in reusable `src/` modules.
4. Evaluate on held-out real embeddings from frozen encoders (e.g., DINO and BioCLIP); protect group/split integrity.
5. Record seeds, dataset version, sample counts, preprocessing, metrics, confounds, and negative results.
6. Store result summaries in experiment-specific notes; avoid committing datasets and large embeddings.

## Papers studied
- Mensink et al. (2013), *Distance-Based Image Classification: Generalizing to New Classes at Near-Zero Cost.*
- Ibrahim et al. (2023), *Newer Is Not Always Better: Rethinking Transferability Metrics, Their Peculiarities, Stability and Performance.*
- Cao et al. (ICML 2026), *On the Power of Statistics in Class-Incremental Learning with Pretrained Models.*

## Status
Initial research scaffold. Folder names are planned experiment themes, **not completed results**.
