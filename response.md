# Coding assistant response — Feature-Geometry

## 1. Task / status

**Requested by the author in chat (2026-11-06):** first add the PyTorch Lightning utilities that every experiment will share. There must be no dead code, and each piece goes in its own module. `task.md` was not updated for this request. **Status: completed.** It contains no experiment, dataset or method.

## 2. Changes

New package `src/lightning_utils/`, one responsibility per file:

| File | Content | Source in `Lightning_Aqua_Vision_Language_v1` (dev) | Changed and why |
|---|---|---|---|
| [seeding.py](src/lightning_utils/seeding.py) | `seed_everything(seed)` = `pl.seed_everything(seed, workers=True)` | `src/aquamonitor_clip/train.py` line 336 | Seed is passed explicitly; the source read the `GLOBAL_SEED` environment variable |
| [run_dir.py](src/lightning_utils/run_dir.py) | `make_run_dir(root, experiment, timestamp)`: `root/experiment/timestamp`. `write_resolved_config(run_dir, config)`: `config_resolved.json`, rank 0 only (`RANK`, then `SLURM_PROCID`) | `callbacks.py` `create_callbacks` (folder) and `train.py` lines 657–684 (shared timestamp, resolved config) | The experiment name is given by the caller instead of being built from Aqua settings; config is a plain dict |
| [callbacks.py](src/lightning_utils/callbacks.py) | `EpochMetricsWriter` (unchanged behaviour). `standard_callbacks(run_dir, monitor, mode, save_top_k, patience)`: best checkpoint(s) by a validation metric, `last.ckpt`, per-epoch metrics, and early stopping only if `patience` is given | `callbacks.py` `EpochMetricsWriter`, `create_callbacks` | Monitored metric is a parameter (source: Aqua F1 at the finest taxonomy level); metrics file always on (source: config flag) |
| [trainer.py](src/lightning_utils/trainer.py) | `build_trainer(callbacks, max_epochs, accelerator, devices, precision, **overrides)` with the source defaults: `deterministic=True`, `min_epochs=1`, `gradient_clip_val=1.0`, `logger=False`, each overridable | `train.py` lines 724–744 | Defaults precision to `32-true`; DDP setup and batch limits are not built in (passed as overrides when needed) |
| [tests/test_lightning_utils.py](tests/test_lightning_utils.py) | One real CPU fit/validate/test cycle on seeded synthetic two-Gaussian data (DataModule and LightningModule are test fixtures only) | — | — |

[README.md](README.md): one line in the layout for `src/lightning_utils/`.

**Deliberately not migrated, because nothing uses them yet:**
- `LinearWarmup`, `LRRangeTest`;
- snapshot checkpoints every N epochs;
- DDP `find_unused_parameters`;
- the print-only `CustomCallback`;
- the local/cloud config switch;
- Aqua-specific checkpoint names.

They can be added when an experiment needs them.

## 3. Validation (run on the Dell, `taxonomist` env: torch 2.6.0, pytorch-lightning 2.6.1, pytest 9.0.3)

`python -m pytest -q`: **8 passed** (5 new and the 3 existing Gaussian tests). The new tests check:
- **Full cycle:** `last.ckpt` and one `*_best.ckpt` are written; `epoch_metrics.jsonl` has epochs 0, 1, 2 with `train_loss`, `val_acc` and `lr`; testing from the best checkpoint gives `test_acc` > 0.9.
- **Seeding:** the same seed gives identical weights.
- **Early stopping:** off without `patience`; with a learning rate of 0 and `patience=1` it stops after epoch 1.
- **Trainer:** the defaults hold and an override replaces them.
- **Settings file:** written by rank 0 only.

**Corrections made on the way:**
- The first run had 1 failure, in my test data rather than the utilities: 12 SGD steps on classes 3 apart reached test accuracy 0.656. I moved the classes 6 apart and raised the test learning rate to 0.5.
- No linter is installed, so I checked imports by hand: every import and every function is used by the tests.

## 4. Scientific interpretation

Infrastructure only, with no research result. Multi-process (DDP) behaviour is not tested beyond the rank-0 environment check.

## 5. Open decisions

1. Which first experiment should use these utilities (the proposed Gaussian-replay task in `task.md`)?
2. Pin the dependency versions in `pyproject.toml` to the tested ones, or keep the current lower bounds?

**Stopped.**
