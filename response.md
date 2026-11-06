# Coding assistant response — Feature-Geometry

**Session of 2026-11-06.** All requests came from the author in chat. The author's `task.md` update 3ad7736 (official benchmark splits only) was read and is followed; I did not change `task.md`. Everything below is complete and stopped for review. Nothing was trained, and no experiment, dataset download or GPU job was run.

## 1. Task / status

| # | Request | Status |
|---|---|---|
| 0 | Freeze `Lightning_Aqua_Vision_Language_v1` and continue in this repository | Done: v1 `response.md` and ledger record the freeze (v1 `dev` 6672eae); its code and results are unchanged |
| 1 | Clone this repository on the Dell | Done: `~/Documents/Feature-Geometry`, branch `main` |
| 2 | Add the PyTorch Lightning utilities common to all experiments; no dead code; one module per piece | Done, commit 0ea7174 |
| 3 | Use the same package versions as on LUMI | Done, commits 4cd7c7f and c954814 |
| 4 | Create a data folder `Feature_geometry_data` in the Research workspace | Done: `~/Documents/Research/Feature_geometry_data` (local, outside git) |
| 5 | Check how the original AquaMonitor parquet files are divided | Done: read-only inspection of all three files, section 4 |
| 6 | Copy three papers from the HP laptop | Done: the author ran the `scp`; arrival checked |
| 7 | Official benchmarks only, no synthetic splits; AquaMonitor paper and classification file added | Rule recorded (data README, shared memory) and matching `task.md` 3ad7736; all three files checked against the paper's Table 4, section 4 |

## 2. Changes

### 2a. Shared Lightning utilities (`src/lightning_utils/`, commit 0ea7174)

| File | Content | Source in v1 (`dev`) | Changed and why |
|---|---|---|---|
| [seeding.py](src/lightning_utils/seeding.py) | `seed_everything(seed)` = `pl.seed_everything(seed, workers=True)` | `src/aquamonitor_clip/train.py` line 336 | Seed passed explicitly; v1 read the `GLOBAL_SEED` environment variable |
| [run_dir.py](src/lightning_utils/run_dir.py) | `make_run_dir(root, experiment, timestamp)` creates `root/experiment/timestamp`. `write_resolved_config(run_dir, config)` writes `config_resolved.json` from rank 0 only (`RANK`, then `SLURM_PROCID`) | `callbacks.py` `create_callbacks`; `train.py` lines 657–684 | The caller gives the experiment name instead of it being built from Aqua settings; config is a plain dict |
| [callbacks.py](src/lightning_utils/callbacks.py) | `EpochMetricsWriter` (same behaviour). `standard_callbacks(run_dir, monitor, mode, save_top_k, patience)` gives best checkpoint(s) by a validation metric, `last.ckpt`, per-epoch metrics, and early stopping only if `patience` is given | `callbacks.py` | The monitored metric is a parameter (v1 used Aqua F1 at the finest taxonomy level); the metrics file is always on (v1 used a config flag) |
| [trainer.py](src/lightning_utils/trainer.py) | `build_trainer(callbacks, max_epochs, accelerator, devices, precision, **overrides)` with v1 defaults: `deterministic=True`, `min_epochs=1`, `gradient_clip_val=1.0`, `logger=False`, each overridable | `train.py` lines 724–744 | Precision defaults to `32-true`; DDP and batch limits are passed as overrides when needed |
| [tests/test_lightning_utils.py](tests/test_lightning_utils.py) | One real CPU fit/validate/test cycle on small mock tensors | — | — |

**The test's mock tensors** (two seeded 2D Gaussian classes) are a unit-test fixture only. This is the exception `task.md` 3ad7736 allows, and they are not a research dataset.

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
| `README.md` | Contents, sha256 checksums, the official-benchmark rule, the check against the paper and the split summary |
| `Aqua_Paraquete_files/aquamonitor-monitor.parquet.gzip` | Original monitor file from the dataset authors; sha256 `28d824f1…`, byte-identical to `~/Documents/aqua_data/aquamonitor-monitor.parquet.gzip` |
| `Aqua_Paraquete_files/aquamonitor-classif.parquet.gzip` | Original classification file from the dataset authors; sha256 `df18b019…`; differs from the derived `aqua_data/aquamonitor-classif-with-taxonomy_all.parquet.gzip` |
| `Aqua_Paraquete_files/aquamonitor-fewshot.parquet.gzip` | Original few-shot file from the dataset authors; sha256 `1fe2a66c…`, new on the Dell |
| `Aqua_Paraquete_files/Papers/aqua_paper/` | AquaMonitor dataset paper (PDF, Markdown text, OOD-class notes): the source of the official benchmarks |
| `Aqua_Paraquete_files/Papers/` | Mensink et al. 2013 (15 pages), Ibrahim et al. (24 pages), Cao et al. (17 pages): the three papers in this README's "Papers studied" |
| `scripts/inspect_aqua_splits.py` | Read-only split inspection of the three files |
| `notes/aqua_splits_summary.txt` | Its saved output; every number in section 4 comes from it, except those quoted from the paper |

The folder stays outside the repository so that data is never committed. Code will receive its path as a setting. The monitor and few-shot files were copied again with the classification file; their checksums are unchanged.

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
- The first test run had 1 failure, in my test fixture rather than the utilities: 12 SGD steps on classes 3 apart reached test accuracy 0.656. I moved the classes 6 apart and raised the test learning rate to 0.5.
- No linter is installed, so imports were checked by hand.

**Data numbers:**
- 154 numbers in the data-folder README were compared with `notes/aqua_splits_summary.txt` (thousands commas removed).
- The only ones not found are quoted from the paper: section numbers, the date, and the paper's few-shot specimen counts 487 / 60 / 145.

**My earlier error:** in the previous `response.md` I counted classes by the `taxon` column (e.g. 122 / 59 / 108 for monitor). The paper's official class is `taxon_group`. Those counts are now labelled as `taxon` counts, and the official counts are in section 4a.

## 4. The AquaMonitor benchmarks (observed)

### 4a. Official benchmarks: paper vs files

The paper (Impiö et al., Sec. 3.1, 4.1, A.3, Table 4) defines three benchmarks, all with class = `taxon_group`:

| Benchmark | Split in the file | Paper evaluation | Match with paper Table 4 (`fold0`) |
|---|---|---|---|
| **Monitoring** (main) | monitor `fold0`: all 2021 train; 2022 val = random 10 % of 2022 specimens; rest of 2022 test | ID classification on 85 classes (frame logits averaged per specimen); OOD detection of 24 classes (entropy, MaxLogit, Energy) | **Exact.** 152 classes; 128 / 60 / 109; 85 shared + 24 OOD (72 specimens, 2,883 images); images 1,640,936 / 110,207 / 1,005,521; specimens 22,882 / 2,028 / 18,279 |
| **Classification** | classif `fold0`–`fold4`: both years, 42 classes with ≥50 specimens, closed set | Main results on the first fold | **Exact.** Images 1,882,046 / 282,923 / 543,054; specimens 29,346 / 4,386 / 8,433; 42 classes in every split |
| **Few-shot** | few-shot `fold0`–`fold4`: 47 classes | 5-NN with cosine distance on image embeddings; test sets of the 5 folds pooled (jackknife) | **Classes and images exact** (47 / 32 / 47; 30,575 / 4,426 / 8,382). **Specimens do not match:** the file has 621 / 93 / 179 distinct `individual` (649 / 97 / 185 `imaging_run`); the paper reports 487 / 60 / 145. Unexplained |

### 4b. Split structure

**All files:** one row per image; `individual` is the specimen; no specimen is in more than one split of any split column.

**Monitor:**
- Temporal: train is all 2021 specimens (bulk21 19,850, dna 1,518, biomass 1,514); val and test are 2022 bulk22.
- Val and test share all 29 sites, all 13 lakes and 296 of val's 309 sample labels.
- `taxon_group_train` and `taxon_group_test` replace rare taxa with `Other` for 72 and 496 specimens.
- A sample label is taxon code plus site with no year, so the 253 labels shared between train and test are the same taxon at the same site in different years.

**Classification:**
- It has the same 2,756,664 images as the monitor file; 48,641 images are marked `'0'` (outside the benchmark) in each fold.
- 5-fold cross-validation by specimen: each of the 42,165 benchmark specimens is in test in exactly one fold.
- Each class has 50–10,643 specimens (median 186.5).
- Years are mixed: 2021 has 22,268 specimens and 2022 has 19,897.
- Train and test share all 50 sites and 22 lakes.

**Few-shot:**
- It is a subset of the monitor file, and shares no specimens and no classes with the classification benchmark.
- 5-fold cross-validation by specimen: each specimen is in test in exactly one fold.
- Val is drawn from the non-test specimens: 507 specimens are never in val, 316 once, 61 twice and 9 three times.
- Years are mixed: 2021 has 523 specimens and 2022 has 370.
- Train and test share 41–43 sites and 20–21 lakes per fold.

## 5. Interpretation and limits

**Infrastructure:**
- It is infrastructure only, with no research result.
- The multi-process (DDP) path is not tested beyond the rank-0 check.
- The tests have not been run inside the LUMI container.
- The local Python patch version and torch build still differ from LUMI.

**Benchmarks:**
- They are used as published. The paper itself stratifies classification and few-shot folds by site, and does not group them.
- Shared sites between val and test (and, for classification and few-shot, between train and test) are an **official-protocol limitation to report, not a reason to re-split** (`task.md` 3ad7736).
- The few-shot specimen-count mismatch must be understood before any few-shot result is compared with the paper.

## 6. Open decisions

1. **Old wording that conflicts with rule 3ad7736:**
   - [README.md](README.md) still says "Synthetic-data tests verify mathematical behavior" and "Derive expected behavior on a simple synthetic dataset first", and lists `experiments/03_gaussian_feature_replay/` as "Gaussian synthetic feature sampling".
   - `src/statistics/gaussian.py` and `tests/test_gaussian.py` are the exploratory code that `task.md` says is not approved for experiments; only their own tests use them.
   - Should the README wording be aligned with the rule, and should that code be kept, or removed under the no-dead-code rule? I changed neither.
2. **Few-shot specimen mismatch:** look into the 621 / 93 / 179 against 487 / 60 / 145 difference now, or only if the few-shot benchmark is chosen?
3. **First benchmark** (`task.md` proposed next task): monitoring, classification or few-shot?
4. Run the 8 tests once inside the LUMI container (a small CPU job of a few minutes)?

**Stopped.**
