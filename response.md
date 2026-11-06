# Coding assistant response — Feature-Geometry

**Session of 2026-11-06.** All requests came from the author in chat; `task.md` was not changed. Everything below is complete and stopped for review. Nothing was trained, and no experiment, dataset download or GPU job was run.

## 1. Task / status

| # | Request | Status |
|---|---|---|
| 0 | Freeze `Lightning_Aqua_Vision_Language_v1` and continue in this repository | Done: v1 `response.md` and ledger record the freeze (v1 `dev` 6672eae); its code and results are unchanged |
| 1 | Clone this repository on the Dell | Done: `~/Documents/Feature-Geometry`, branch `main` |
| 2 | Add the PyTorch Lightning utilities common to all experiments; no dead code; one module per piece | Done, commit 0ea7174 |
| 3 | Use the same package versions as on LUMI | Done, commits 4cd7c7f and c954814 |
| 4 | Create a data folder `Feature_geometry_data` in the Research workspace | Done: `~/Documents/Research/Feature_geometry_data` (local, outside git) |
| 5 | Check how the original AquaMonitor parquet files are divided | Done: read-only inspection, section 4 |
| 6 | Copy three papers from the HP laptop | Done: the author ran the `scp`; arrival checked |

## 2. Changes

### 2a. Shared Lightning utilities (`src/lightning_utils/`, commit 0ea7174)

| File | Content | Source in v1 (`dev`) | Changed and why |
|---|---|---|---|
| [seeding.py](src/lightning_utils/seeding.py) | `seed_everything(seed)` = `pl.seed_everything(seed, workers=True)` | `src/aquamonitor_clip/train.py` line 336 | Seed passed explicitly; v1 read the `GLOBAL_SEED` environment variable |
| [run_dir.py](src/lightning_utils/run_dir.py) | `make_run_dir(root, experiment, timestamp)` creates `root/experiment/timestamp`. `write_resolved_config(run_dir, config)` writes `config_resolved.json` from rank 0 only (`RANK`, then `SLURM_PROCID`) | `callbacks.py` `create_callbacks`; `train.py` lines 657–684 | The caller gives the experiment name instead of it being built from Aqua settings; config is a plain dict |
| [callbacks.py](src/lightning_utils/callbacks.py) | `EpochMetricsWriter` (same behaviour). `standard_callbacks(run_dir, monitor, mode, save_top_k, patience)` gives best checkpoint(s) by a validation metric, `last.ckpt`, per-epoch metrics, and early stopping only if `patience` is given | `callbacks.py` | The monitored metric is a parameter (v1 used Aqua F1 at the finest taxonomy level); the metrics file is always on (v1 used a config flag) |
| [trainer.py](src/lightning_utils/trainer.py) | `build_trainer(callbacks, max_epochs, accelerator, devices, precision, **overrides)` with v1 defaults: `deterministic=True`, `min_epochs=1`, `gradient_clip_val=1.0`, `logger=False`, each overridable | `train.py` lines 724–744 | Precision defaults to `32-true`; DDP and batch limits are passed as overrides when needed |
| [tests/test_lightning_utils.py](tests/test_lightning_utils.py) | One real CPU fit/validate/test cycle on seeded synthetic two-Gaussian data; its DataModule and LightningModule exist only as test fixtures | — | — |

**Not migrated, because nothing uses them yet:**
- `LinearWarmup` and `LRRangeTest`;
- snapshot checkpoints every N epochs;
- DDP `find_unused_parameters`;
- the print-only `CustomCallback`;
- the local/cloud config switch;
- Aqua-specific checkpoint names.

### 2b. LUMI package versions (commits 4cd7c7f, c954814)

**Source of the versions:** package metadata read inside the LAIF container that every v1 LUMI job used, `lumi-multitorch-full-u24r70f21m50t210-20260731_122833.sif`. It was one login-node command, metadata only, with no torch import and no job. The v1 extra-package folder (`/scratch/project_462001534/aqua_v1/pkgs`: open_clip, ftfy, regex, imbalanced-learn, sklearn_compat) replaces none of these.

| Package | LUMI container | Pinned in [pyproject.toml](pyproject.toml) | Local `.venv` | Before |
|---|---|---|---|---|
| Python | 3.12.3 | `>=3.12,<3.13` | 3.12.12 | ≥3.10 |
| torch | 2.10.0 (ROCm 7.0 build) | 2.10.0 | 2.10.0+cpu | ≥2.1 |
| pytorch-lightning | 2.6.5 | 2.6.5 | 2.6.5 | ≥2.2 |
| numpy | 2.3.5 | 2.3.5 | 2.3.5 | ≥1.24 |
| scipy (needed by scikit-learn) | 1.18.0 | 1.18.0 | 1.18.0 | not listed |
| scikit-learn | 1.9.0 | 1.9.0 | 1.9.0 | ≥1.3 |
| matplotlib | 3.11.1 | 3.11.1 | 3.11.1 | ≥3.7 |
| pytest | 9.1.1 | 9.1.1 (dev) | 9.1.1 | ≥7.4 |

**Other changes:**
- `jupyter>=1.0` is left as it was, because it is not in the container.
- [README.md](README.md): the setup section now uses Python 3.12 and CPU torch locally, and the container on LUMI; the layout gains one line for `src/lightning_utils/`.
- `.gitignore`: adds `*.egg-info/`.
- The local `.venv` (git-ignored, 1.1 GB) was built with `uv`, with CPU torch from the PyTorch index.

### 2c. Data folder (local, not in git)

`~/Documents/Research/Feature_geometry_data/`:

| Path | Content |
|---|---|
| `README.md` | Contents, sha256 checksums and the split summary below |
| `Aqua_Paraquete_files/aquamonitor-monitor.parquet.gzip` | Original monitor file from the dataset authors; sha256 `28d824f1…`, byte-identical to `~/Documents/aqua_data/aquamonitor-monitor.parquet.gzip` |
| `Aqua_Paraquete_files/aquamonitor-fewshot.parquet.gzip` | Original few-shot file from the dataset authors; sha256 `1fe2a66c…`, new on the Dell (no derived copy matches it) |
| `Aqua_Paraquete_files/Papers/` | Mensink et al. 2013 (15 pages), Ibrahim et al. (24 pages), Cao et al. (17 pages): the three papers in this README's "Papers studied" |
| `scripts/inspect_aqua_splits.py` | Read-only split inspection script |
| `notes/aqua_splits_summary.txt` | Its saved output; every number in section 4 comes from it |

The folder stays outside the repository so that data is never committed. Code will receive its path as a setting.

## 3. Validation

**Tests:**
- `.venv/bin/python -m pytest -q`: **8 passed**, with the LUMI-pinned versions (5 new tests and the 3 existing Gaussian tests). They check:
  - checkpoints and the per-epoch metrics file are written;
  - testing from the best checkpoint gives `test_acc` > 0.9;
  - the same seed gives identical weights;
  - early stopping is on only when a patience is given;
  - Trainer defaults hold and can be overridden;
  - the settings file is written by rank 0 only.
- They passed earlier with torch 2.6.0, Lightning 2.6.1 and Python 3.11, before the pins.

**Corrections on the way:**
- The first test run had 1 failure, in my test data rather than the utilities: 12 SGD steps on classes 3 apart reached test accuracy 0.656. I moved the classes 6 apart and raised the test learning rate to 0.5.
- No linter is installed, so imports were checked by hand.

**Data numbers:**
- 65 numbers in the data-folder README were compared with `notes/aqua_splits_summary.txt` (thousands commas removed), and all were found.
- The first check missed one: the few-shot 2021 count 523 had been worked out by hand (512 + 11). The script now prints it.

## 4. How the AquaMonitor files are divided (observed)

**Both files:** one row per image; `individual` is the specimen; no specimen is in more than one split of any split column.

**Monitor file:** 2,756,664 images, 43,189 specimens, 145 taxa; split column `fold0`. This is the temporal split used in the earlier Aqua work.

| `fold0` | Year | Source | Images | Specimens | Taxa |
|---|---|---|---|---|---|
| train | 2021 | bulk21 19,850, dna 1,518, biomass 1,514 specimens | 1,640,936 | 22,882 | 122 |
| val | 2022 | bulk22 | 110,207 | 2,028 | 59 |
| test | 2022 | bulk22 | 1,005,521 | 18,279 | 108 |

- **Val and test are not separated by place.** They share all 29 sites, all 13 lakes and 296 of val's 309 sample labels.
- **Taxa:**
  - 23 test taxa are not in train (the open-set part);
  - 37 train taxa are not in test;
  - 3 val taxa are not in train.
- **Rare taxa:** `taxon_group_train` and `taxon_group_test` replace rare taxa with `Other` for 72 and 496 specimens.
- **Sample labels:** a label is taxon code plus site, with no year. So the 253 labels shared between train and test are the same taxon at the same site in different years, not the same physical sample.

**Few-shot file:** 43,383 images, 893 specimens, 46 taxa; split columns `fold0`–`fold4`.
- **Subset:** every image and specimen is also in the monitor file.
- **Folds:** 5-fold cross-validation by specimen. Each specimen is in test in exactly one fold, and each fold has about 621 train / 93 val / 179 test specimens.
- **Val:** drawn from the non-test specimens. 507 specimens are never in val, 316 once, 61 twice and 9 three times.
- **Taxa:** all 46 are in train and test of every fold; val has 32–37. Each taxon has 5–44 specimens (median 16.5).
- **Not temporal:** 2021 (523 specimens) and 2022 (370) are mixed.
- **Not separated by place:** train and test share 41–43 sites, 20–21 lakes and 82–91 sample labels per fold.

## 5. Interpretation and limits

**Infrastructure:**
- It is infrastructure only, with no research result.
- The multi-process (DDP) path is not tested beyond the rank-0 check.
- The tests have not been run inside the LUMI container.
- The local Python patch version (3.12.12 against 3.12.3) and torch build (CPU against ROCm) still differ from LUMI.

**Data:**
- **Monitor `fold0`:** suits a temporal question (2021 → 2022, with new taxa in 2022). Choices made on val are not location-independent of test.
- **Few-shot file:** closed-set; specimens are separate, but year and place are not.
- **Shared limit:** neither file separates sites. Any experiment that needs place independence has to define its own grouping, and that would need its own approved task.

## 6. Open decisions

1. Run the 8 tests once inside the LUMI container? This would be a small CPU job of a few minutes.
2. Which experiment should use these utilities first? `task.md` proposes Gaussian replay on synthetic data.
3. When real AquaMonitor features are needed: monitor `fold0`, few-shot folds, or both? And should the inspection script move into this repository (pandas and pyarrow would then need pinning to LUMI versions)?

**Stopped.**
