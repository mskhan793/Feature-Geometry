# Coding assistant response — Feature-Geometry

**Current task:** `task.md` d57a630, "Bounded eight-dataset × four-foundation-model frozen-feature extraction".
**Status: 32 of 32 combinations have complete, checked train+val feature caches. Stopped for author review.**
- **Not done (not authorised):** no test images encoded for new combinations, no classifiers, no GBC/FDA, no test
  scoring, no taxonomy pooling.
- **Evidence:** `docs/evidence/benchmark_readiness/` (README); extraction in `extraction_c2/` (README).
- **Standard:** `docs/benchmark_standard.md` (§6 new: extraction protocol).
- **Previous response** (readiness, task 8538df5): commit 86d5636.

## 1. Completion (train+val, 8 × 4)

| Dataset | DINOv3-L | DINOv2-L | BioCLIP 2 | SigLIP 2 |
|---|---|---|---|---|
| FIN-Benthic2 | complete (reused) | complete (new) | complete (new) | complete (new) |
| AquaMonitor | complete (reused) | complete (new) | complete (new) | complete (new) |
| BIOSCAN-5M | complete (reused, v1) | complete (new; v1 cache not equivalent) | complete (reused, v1) | complete (new) |
| DF24 | complete (new) | complete (new) | complete (new) | complete (new) |
| Food-101 | complete (new) | complete (new) | complete (new) | complete (new) |
| Oxford-IIIT Pet | complete (new) | complete (new) | complete (new) | complete (new) |
| CIFAR-100 | complete (new) | complete (new) | complete (new) | complete (new) |
| FGVC-Aircraft | complete (C1) | complete (C1) | complete (C1) | complete (C1) |

- **Counts:** 24 new, 4 reused, 4 from the C1 pilots. Pending: none. Blocked: none.
- **"Complete"** means: the C3 record passes all checks (§5) and, for caches made here, the cache verification passes.
- **Per-combination records:** `extraction_c2/records/<encoder>__<dataset>.json`.
- **Rows per combination** (same for every encoder):

  | Dataset | Images | Units |
  |---|---|---|
  | FIN-Benthic2 | 186,284 | 9,631 specimens |
  | AquaMonitor | 836,159 | 42,165 specimens |
  | BIOSCAN-5M | 303,960 | 303,960 records |
  | DF24 | 266,273 | 159,026 observations |
  | Food-101 | 75,750 | 75,744 content units |
  | Pet | 3,680 | 3,680 |
  | CIFAR-100 | 50,000 | 49,986 |
  | Aircraft | 6,667 | 6,667 |

## 2. C0 — data, packaging, reuse decisions

- **Download** (job 23690625):
  - DF24 300-px images: `DF20-300px.tar.gz`, 6,422,709,063 bytes, sha256 b7b57217…, gzip test passed, 295,938
    images;
  - DF24 metadata zip: checksum OK;
  - SigLIP 2 so400m-patch16-256 weights: sha256 equal to the Hugging Face LFS record.
- **Packing** (jobs 23691093–23691097; `c0/<dataset>_trainval.json`): train+val images into byte-verified HDF5 on
  `/flash`, test not packed.

  | Dataset | Train / val images | HDF5 bytes | Content hashes rechecked |
  |---|---|---|---|
  | Food-101 | 68,174 / 7,576 | 3,851,174,728 | yes |
  | Pet | 3,311 / 369 | 378,523,784 | yes |
  | Aircraft | 3,334 / 3,333 | 1,839,385,016 | yes |
  | BIOSCAN-5M | 289,203 / 14,757 | 2,473,410,632 | no content table; read-back verified |
  | DF24 | 239,727 / 26,546 | 6,114,597,824 | yes |

- **DF24 licence: correction of my earlier report.**
  - The "Public Domain vs CC BY-NC-ND 4.0" conflict I attributed to DF24 belongs to NAFlora-1M (v1 dataset audit,
    NAFlora section).
  - The DF24 dataset authors' README (commit c5c19b41, copy in `df24_licence/`) says BSD, with data, metadata and
    models available for non-commercial research only.
  - The DF24 metadata CSV has no licence column, only a per-image `rightsHolder` (attribution), so I found no
    conflicting DF24 statement. The use here is non-commercial research.
- **FGVC-Aircraft:** provisionally accepted with the documented MD5 caveat (two byte-identical downloads,
  `archives.tsv`). Packing used the archive's own split files, and the content hashes were rechecked. Nothing in
  packing or extraction disagreed with them. I did not compare the split files with any third copy.
- **Duplicates** (`duplicates/`; metadata only, nothing removed):
  - CIFAR-100: 26 exact-duplicate groups (52 images); 10 cross train+val / test, 17 have conflicting labels.
  - Food-101: **8 groups (16 images)**, not 16 groups as `task.md` says; 2 cross, 2 have conflicting labels.
  - No group is split between train and val.
  - Candidate rules for your decision (§8): R1 flags 10 / 2 test images; R2 flags 24 / 4 train+val images.

## 3. C1 — pilots and reuse audit

- **v1 BIOSCAN-5M caches vs fresh extraction** (512 fixed train images; tolerance fixed beforehand: relative L2 ≤ 1e-4,
  cosine ≥ 0.999999):
  - DINOv3-L passes (max 1.5e-5), so it is reused;
  - BioCLIP 2 passes (3.3e-5), so it is reused;
  - **DINOv2-L fails** (max 4.9e-4, min cosine 0.99999988, same checkpoint, transform and dimension). Its
    BIOSCAN-5M train+val was re-extracted. The v1 cache is left untouched.
- **Repeat extraction:** 512 of 512 rows bitwise identical for each of the four encoders.
- **Throughput** (Aircraft, images/s per GCD): DINOv3-L 104.8, DINOv2-L 117.2, BioCLIP 2 133.3, SigLIP 2 76.5.
- **Peak GPU memory:** 2.15–2.71 GB.
- **Error (mine):** the first 3 BioCLIP 2 pilot jobs failed at once (`open_clip` is not in the container). Fixed by
  putting the verified add-on folder on `PYTHONPATH` (f74bcb3); rerun passed.

## 4. C2 — extraction

- **Plan from the measured rates** (`c2_plan/`):
  - A first version estimated 32 GPU-h, because GPU nodes would have sat idle during the CPU checks.
  - I moved verification and the 256-row CPU-vs-GPU check into a chained CPU job (`check_extraction.sh`).
  - New estimate: 11.2 GPU-h for the 24 new combinations (task estimate 11.5).
- **Batch 1:** FIN-Benthic2, Food-101, Pet, CIFAR-100; 15 GPU jobs, 23696935–23696964.
- **Batch 2:** AquaMonitor, BIOSCAN-5M, DF24; 9 GPU jobs, 23697108–23697124. Submitted after batch 1 had been
  inspected.
- **Checks:** every GPU job and every CPU check completed. All caches verified complete with no problems, all
  values finite.
- **CPU vs GPU (first 256 rows):**
  - max relative norm difference ≤ 1.5e-5, except DF24 × DINOv2-L at 1.09e-4 (max abs 0.023);
  - DINOv2-L has the largest differences on every dataset;
  - no threshold was set for this check, so I report it rather than call it a failure.
- **Throughput at scale** (images/s per GCD), e.g. AquaMonitor: BioCLIP 2 150.6, DINOv2-L 135.4, SigLIP 2 94.4.
- **No chunk failed,** so nothing was rerun.

## 5. C3 — quality assurance

Every one of the 32 records (`records/`, array 23697582) passes:
- **Identity:**
  - single-split datasets: ids equal the standardised manifest's available train+val ids, and no test id is present;
  - FIN-Benthic2 / AquaMonitor: every id is in the standardised manifest, and all specimens are present.
- **Units:** the present units equal the expected units (table in §1).
- **Features:** float32, all finite, positive norms, sha256 recorded in manifest order.
- **Rank:**
  - DINOv3-L and DINOv2-L: rank 1023 of 1024 on all 8 datasets. The null direction is the final-LayerNorm 1/w
    direction (|cos| 1.0).
  - BioCLIP 2 (768) and SigLIP 2 (1152): full rank on all 8.
- **Tests:** 106 passed locally. New: `tests/test_combination_record.py`. The refactored null-space audit
  reproduces the 2026-11-12 output exactly.

**Test isolation:**
- The new caches hold no test images of the single-split datasets.
- The reused v1 BIOSCAN-5M caches also contain test and open-world folders; only `train` and `val` were opened.
- FIN-Benthic2 and AquaMonitor have no single test set. The cache is the whole frame selection, so each specimen is
  test in exactly one fold and train or val in the other four. As before, the fold's test rows are dropped before
  any feature lookup (`load_fold`).

## 6. Resources (all on project_462001534; `extraction_c2/resources.json`, allocation-based)

| Stage | GPU-h | CPU core-h |
|---|---|---|
| C0 download + packing | 0 | 1.09 + 1.74 |
| C1 pilots + checks | 3.53 | 0.29 |
| C2 GPU extraction (24 jobs) | 8.50 | 0 |
| C2 CPU checks (24 jobs) | 0 | 37.45 |
| C3 records (32 tasks) | 0 | 2.21 |
| **Total** | **12.03** | **42.78** |

- **GPU-hours:** C2 extraction used 8.50 GPU-h against the 11.2 planned. With the pilots, 12.03 against the task's
  11.5.
- **Storage:** new feature caches 20,967,569,244 bytes on `/scratch`; train+val HDF5 packs 14,943,683,535 bytes on
  `/flash`; source archives 17,972,400,070 bytes.
- **Energy:** Slurm's energy counter adds up to an implausibly low value for these GPU nodes, so I do not report an
  energy figure.
- **Quota:** `lumi-allocations` still showed its cached pre-C2 value.

## 7. Code (commits a6ff351, 937cd61, and this one)

- **New:**
  - `scripts/benchmark/c2_plan.py`, `combination_record.py`, `completion_matrix.py`, `duplicate_groups.py`,
    `sum_resources.py`;
  - `scripts/lumi/check_extraction.sh`, `submit_c2.sh`, `c3_records.sh`;
  - `src/features/null_space.py`, `src/features/v1_bioscan_cache.py`.
- **Changed:**
  - `extract_encoder.sh` no longer runs the checks on the GPU node;
  - `audit_feature_null_space.py` and `compare_v1_bioscan_cache.py` now import the shared modules (output
    unchanged);
  - `fetch_sources.sh` also fetches the DF24 README.

## 8. Decisions for the author

1. **Duplicates:** apply R1 and/or R2 (§2) in later evaluation, or keep them only reported?
2. **DINOv2-L v1 BIOSCAN-5M cache:** superseded by the new extraction. Keep it (my default; nothing deleted), or
   may it be removed?
3. **DF24 terms:** non-commercial research only. Please confirm that this fits the intended use, including any
   later sharing of derived features.
4. **Five-fold datasets:** confirm that one fold-independent cache per encoder (test rows dropped per fold at
   analysis time) is the intended "train+val only" reading for FIN-Benthic2 and AquaMonitor.
5. **Next task:** the caches are ready for a separately approved analysis. Nothing further will run until then.

## 9. End
