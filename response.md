# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 9d732ba, "Prepare AquaMonitor Classification frozen features". It was extended by:
- 17f864f and 442999a: a nine-encoder shortlist, a feasibility audit for the six new candidates, and the long-term research vision, which authorises no experiments;
- 8ac12a7: authorises the three-baseline LUMI smoke test (cap: 1 node, 1 h, about 4 GPU-hours) and asks for one recommendation per candidate;
- 4405baa: sets the official-checkpoint resolution policy.

**Status: partial, stopped before any GPU job.**
- **Done:** data checks, manifest and integrity audit, extraction code for the three baselines, local tests, and the **metadata feasibility audit of the six new candidates** (section 2e).
- **Not yet run:** the LUMI smoke test, which is prepared and tested locally but **blocked because LUMI is in a maintenance break**: its login node says so and then closes the connection (section 4). So there is no v1 equivalence result, no throughput and no estimate yet.
- **Done now:** one recommendation per new candidate (section 4b).
- **Not started:** the full extraction. It waits for that estimate and the author's approval, as `task.md` requires.

Section 1 covers the earlier work of 2026-11-06, kept briefly because it is in previous commits.

## 1. Earlier work (2026-11-06, before task 9d732ba)

| Item | Result | Commit |
|---|---|---|
| Freeze `Lightning_Aqua_Vision_Language_v1`, continue here | v1 code and results unchanged (v1 `dev` 6672eae) | — |
| Shared Lightning utilities `src/lightning_utils/` (seeding, run folder, callbacks, Trainer defaults) | 5 tests | 0ea7174 |
| Package versions pinned to the LUMI LAIF container `lumi-multitorch-full-u24r70f21m50t210-20260731_122833` | local `.venv` (Python 3.12, CPU torch) | 4cd7c7f, c954814 |
| Data folder `~/Documents/Research/Feature_geometry_data` (outside git; its README lists every file with sha256) | original monitor, classification, few-shot parquets; AquaMonitor paper + 3 papers | — |
| Official-benchmarks-only rule (`task.md` 3ad7736) | monitor and classification `fold0` equal the paper's Table 4 exactly. Few-shot classes and images match, but specimens do not (621 / 93 / 179 against the paper's 487 / 60 / 145); this is unexplained. Classes counted by `taxon_group` | 97aaf3f |

## 2. Task 9d732ba: work done

### 2a. Is our self-built image HDF5 correct? Yes, completely

The image HDF5 was built by us earlier, so it was checked against the dataset authors' release before use.

| Check | Result | Evidence |
|---|---|---|
| All 35 metadata columns × 2,756,664 rows of the HDF5 against the original classification parquet; HDF5 labels against `taxon_group`; 3,356 images decoded against their recorded width/height | 0 mismatches; all decode with the recorded size (no shift) | `scripts/audit_aquamonitor_hdf5.py` → `Feature_geometry_data/notes/hdf5_alignment_audit.json` |
| Authors' release downloaded to LUMI: Hugging Face `mikkoim/aquamonitor`, revision `9a30834874d0428be529ab0d35d8aa3a687dd43a`, CC BY 4.0, 132 files (111 image shards), 109.09 GB, no thumbnails | job 23526380 COMPLETED in 13:49; 132/132 files match the published checksums | `data_manifests/aquamonitor_hf_9a30834.tsv`, `scripts/lumi/download_aquamonitor_release.sh`; data in `/scratch/project_462001534/feature_geometry/aquamonitor_hf_9a30834` |
| Every released image against `/flash/project_462001534/data_hdf_aqua/aquamonitor_complete.h5`, by name and bytes | job 23526542 **PASS** in 1:06: 2,756,664 / 2,756,664 byte-identical, every HDF5 position covered exactly once | `scripts/compare_hdf5_with_release_shard.py`, `scripts/lumi/check_hdf5_vs_release.sh` → `notes/hdf5_vs_release_full.json` |
| Our classification parquet against the release | identical sha256 (`df18b019…`) | manifest TSV |

**The first full-check job failed (my error).** Job 23526533 failed after 1 second because I left out the container bindings module, so `/scratch` was not visible inside the container. This is the same mistake as v1 job 23374480.
- **Fix:** every container job now loads the shared `scripts/lumi/container_env.sh` and first tests that its script is readable inside the container (dd584ce).
- **Record:** the rule is now in the LUMI skill and in memory.

### 2b. Row index vs HDF5 position (the earlier "mismatch sometimes")

`scripts/check_parquet_hdf5_index.py` was run over all 24 Aqua parquet files on the Dell (`notes/parquet_hdf5_index_check.json`).
- **Index is safe as the HDF5 position:** the full classification and monitor files (released and derived), and all `monitor-with-hierarchy*` subsets.
- **Index is wrong for every row:** the released few-shot (43,383 rows) and biomass (120,627) files, and our `fewshot-with-taxonomy*` files. Their index was reset, so the position must be found by image name.
- **Not a parquet:** `aquamonitor-monitor.taxonomy.gzip` is a gzipped CSV.

These old files are left unchanged. **Rule here:** a DataFrame index is never used as an HDF5 position. The manifest stores an explicit `hdf5_row` looked up by name, and every image read checks the name stored at that position.

### 2c. Manifest and integrity audit (task step 2)

`scripts/build_aquamonitor_manifest.py` writes `Feature_geometry_data/manifests/aquamonitor_classif_manifest.parquet` (sha256 `d6e63bb6…`, no index) and `….audit.json`.
- **Contents:** benchmark rows only, in the original order; images marked `'0'` in every fold are dropped.
- **Columns:** `img`, `hdf5_row`, `individual`, `imaging_run`, `taxon_group`, `fold0`–`fold4`, `sample`, `site`, `lake`, `year`, `dataset`, `camera`.

| | Images train / val / test | Specimens train / val / test | Classes | Specimens in >1 split |
|---|---|---|---|---|
| fold0 | 1,882,046 / 282,923 / 543,054 | 29,346 / 4,386 / 8,433 | 42 / 42 / 42 | 0 |
| fold1 | 1,883,026 / 277,789 / 547,208 | 29,346 / 4,386 / 8,433 | 42 / 42 / 42 | 0 |
| fold2 | 1,885,562 / 282,836 / 539,625 | 29,346 / 4,386 / 8,433 | 42 / 42 / 42 | 0 |
| fold3 | 1,882,953 / 285,366 / 539,704 | 29,346 / 4,386 / 8,433 | 42 / 42 / 42 | 0 |
| fold4 | 1,886,474 / 283,117 / 538,432 | 29,346 / 4,386 / 8,433 | 42 / 42 / 42 | 0 |

**Totals:** 2,708,023 images, 42,165 specimens, 42 classes; every `hdf5_row` is unique. Each of the 42,165 specimens is in test in exactly one fold. The manifest builder refuses:
- split values other than train / val / test / `'0'`;
- an image marked `'0'` in only some folds;
- duplicate names;
- any image missing from the HDF5.

### 2d. Extraction code (task steps 1, 3–5; uncommitted until this response)

| File | Role | Source in v1 (`dev`) |
|---|---|---|
| `src/data/aquamonitor_manifest.py` | manifest + name → HDF5 position lookup | replaces `aqua_dataset.build_img_to_hdf5_idx` and index-based reads |
| `src/data/hdf5_images.py` | `Hdf5ImageDataset`: lazy per-worker HDF5, name check on every read, any unreadable image raises | `aqua_dataset.AquaMonitorHierarchicalDataset.__getitem__` (image read only). The v1 fallback that returned an all-zero dummy image on error is **not** kept |
| `src/data/image_prediction_datamodule.py` | `LightningDataModule` with `predict_dataloader` only (ordered, no shuffle, no augmentation) | — |
| `src/encoders/frozen_encoders.py` | BioCLIP-2, DINOv2-L, DINOv3-L | `scripts/encode_frozen_backbone.py`, `scripts/run_bioscan.sh` (same model ids, transforms, outputs) |
| `src/lightning_modules/frozen_encoder.py` | `FrozenEncoderModule.predict_step` (eval mode, no gradients; features as float32) | — |
| `src/features/chunk_store.py` | chunked resumable cache with atomic writes, identity check, sha256, norms | replaces v1 shard files `rows_NNN.npy` + `meta_NNN.json` |
| `src/features/extract.py` | per chunk: skip if done, `Trainer.predict`, order check, write | `encode_frozen_backbone.main` loop |
| `scripts/extract_features.py` | CLI, one process per GPU (chunks `task::n_tasks`) | — |
| `scripts/verify_feature_cache.py` | coverage of every manifest row once, checksums, shape, finiteness | `check_feature_cache.py` (idea) |
| `scripts/compare_with_v1_cache.py` | equivalence gate: reproduce the v1 Aqua cache, max abs diff ≤ 1e-4 × max \|v1\| | `run_bioscan.sh` step 1 |

**Encoders (all frozen; raw outputs, not unit-normalised):**

| Name | Model id | Preprocessing | Feature |
|---|---|---|---|
| `bioclip2` | `hf-hub:imageomics/bioclip-2` (open_clip 3.3.0) | open_clip's own eval preprocess | `model.visual(x)` |
| `dinov2_l` | `facebook/dinov2-large` | Resize 256 bicubic → CenterCrop 224 → ImageNet normalisation (classic PIL transforms, as v1) | `pooler_output` (final-norm CLS token) |
| `dinov3_l` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | ToImage → Resize((256, 256)) → ImageNet normalisation, no crop (as v1) | `pooler_output` |

**Where the weights come from:** the v1 Hugging Face cache on LUMI, read offline (`HF_HUB_OFFLINE=1`), so no weights are downloaded or substituted.

**New pins in `pyproject.toml`** (LUMI versions): torchvision 0.25.0, transformers 5.10.1, open_clip_torch 3.3.0.
- open_clip is not in the container; on LUMI it comes from the v1 add-on folder `aqua_v1/pkgs`.
- Indirect packages in the local `.venv` differ slightly from LUMI: huggingface-hub 1.33.0 against 1.26.0, and timm 1.0.31 against 1.0.28.

**`.gitignore`:** `data/` became `/data/`. The old rule also ignored the code folder `src/data/`; a top-level data folder stays ignored.

**Removed while writing:** none of the earlier code. The transform-consistency check in `verify_feature_cache.py` was fixed before its first run: it had read chunk files by list position instead of chunk number.

### 2e. Nine-encoder feasibility audit (`task.md` 17f864f, metadata only)

**Report:** [docs/encoder_feasibility_audit.md](docs/encoder_feasibility_audit.md).

**Evidence:** `docs/evidence/encoder_candidates/`. It holds the Hugging Face API answer, the config files and the model card of every repository at its immutable revision, plus a local interface check. It was produced by `scripts/encoder_candidate_audit.py` and `scripts/check_encoder_support.py`. No weights were downloaded and nothing was run.

| Candidate | Exact repository found | Licence / access | Feature, dim | Blocking question |
|---|---|---|---|---|
| DINOv3-H+ | `facebook/dinov3-vith16plus-pretrain-lvd1689m` @ `c807c9ee` (the revision v1 pinned on LUMI) | DINOv3 licence, gated; v1 has the weights | `pooler_output`; dim to read from the LUMI cache | none |
| SigLIP 2 So400M 384 | `…-patch16-384` @ `dd658faa` and `…-patch14-384` @ `e8e48729` | Apache-2.0 | `get_image_features`, 1152 | which patch size |
| SigLIP 2 Giant | `…-giant-opt-patch16-384` @ `a713301b` and `…-256` @ `46d7129e` | Apache-2.0 | `get_image_features`, 1536 | which resolution |
| PE-Core-G | `facebook/PE-Core-G14-448` @ `a6046680` (needs Meta's `perception_models` code) and Meta's timm port `…_timm` @ `f537e434` | Apache-2.0 | image embedding, 1280 | official code or timm port |
| C-RADIOv4-H | `nvidia/C-RADIOv4-H` @ `0057b339` | NVIDIA Open Model Licence | `summary`, dim not stated | accept the repository's own code; fix one input size |
| AIMv2 | 4 sizes × 3 resolutions (+ native / distilled / LiT) | Apple AMLR | no pooled output (patch tokens only) | which variant; which pooling |

**Known dependence between models** (stated by the model cards):
- C-RADIOv4-H is distilled from SigLIP2-g (the Giant 384 checkpoint above), DINOv3-7B and SAM3.
- DINOv3-L and DINOv3-H+ share a family and corpus.

**Not yet checked:** the LUMI side of the interface check (timm 1.0.28's PE support, cached weights, the DINOv3-H+ config). LUMI logins were closed for a maintenance break on 2026-11-07 (section 4), and the check runs when it is reachable.

**No adapter for the six candidates has been written.** `task.md` asks for them only after feasibility is confirmed and the variant questions are settled.

## 3. Validation actually run

**Unit and integration tests:** `.venv/bin/python -m pytest -q` gives **24 passed**:

| File | Tests | What they cover |
|---|---|---|
| `test_aquamonitor_manifest.py` | 4 | `'0'` rows dropped, folds preserved, `hdf5_row` found by name, missing image and partial-`'0'` rows refused |
| `test_chunk_store.py` | 6 | rows covered once, write then done, other inputs refused, corrupted file refused, partial write not done, wrong shape / non-finite refused |
| `test_feature_extraction.py` | 3 | real `Trainer.predict`, features equal each named image, resume skips done chunks, repeat is identical |
| `test_hdf5_images.py` | 3 | right image per name, wrong position raises, unreadable image raises |
| `test_lightning_utils.py` | 5 | earlier utilities (checkpoints, metrics file, seeding, early stopping, Trainer defaults) |
| `test_gaussian.py` | 3 | existing math utility |

The fixtures are a tiny HDF5 and parquet. In the fixture HDF5 the images are in a different order from the parquet, so a position taken from the index would fail the tests.

**Data checks:** the alignment audit, the release download, the full byte comparison and the index check in sections 2a–2c.

**Encoder audit:** every number and revision in the report checked against the saved evidence. The only values not found literally are deliberate roundings (e.g. 309,197,824 → 309.2 M) or digits inside revision hashes.

**Not run yet:**
- **Any GPU run of the new code** (LUMI blocked, section 4).

**Real-weights run on the Dell** (replacing the earlier stopped attempt): DINOv3-L on CPU, on the first 64 real benchmark images. Extraction, verification, resume and repeat all passed, and the repeat was bitwise identical (section 4).

## 4. Three-baseline LUMI smoke test (`task.md` 8ac12a7): BLOCKED, not submitted

**Blocker: LUMI maintenance break.**
- **What happened:** my SSH attempts on 2026-11-07 (about 8) all ended with "Connection closed by 193.167.209.16x port 22", once during the initial handshake (`kex_exchange_identification`).
- **Cause, found by the author:** the author logged in by hand with the passphrase. After authentication the login node printed "Maintenance break ongoing. For more information see: https://lumi-supercomputer.eu/service-breaks/" and closed the connection. So the key works, and the Dell's address is not blocked: logins are closed for maintenance.
- **Reason, from LUMI's notice to users (shared by the author):** "access to LUMI has been temporarily closed due to recently published Linux kernel vulnerability." The notice gives no reopening date.
- **What's still unknown:** LUMI's service-breaks page lists no break for 7 November. Its latest entry is the urgent kernel-patch maintenance of 2 November, extended to 3 November, which said overlapping jobs wait and jobs started during bring-up are requeued. When this break ends, and whether it affects compute or storage, is not known.

**No job was submitted and no smoke or throughput result exists.** I will submit after one successful login.

**Prepared and tested locally** (commit f5c71a7):
- `scripts/lumi/smoke_baselines.sh` (job):
  - `dev-g`, 1 node, **3 GCDs** (one per encoder, in parallel), 7 CPUs per task, 120 GB, **1 h limit**;
  - at most 3 GCD-hours ≈ **1.5 GPU-hours**, under the 4 GPU-hour cap, on project 462001534;
  - it first checks that the code, HDF5, manifest and v1 parquet are readable inside the container.
- `scripts/lumi/smoke_baselines_task.sh` (per encoder):
  1. **v1 equivalence gate** on the fixed 512 images (first 512 v1 val rows, order checked by v1's image-name hash). It reports max and mean error, the tolerance (1e-4 × max \|v1\|) and pass/fail.
  2. **Extraction** of the first 16,384 manifest rows in 4 chunks of 4,096.
  3. **Verification** (IDs, shape, finiteness, checksums).
  4. **Rerun**, which must skip all 4 chunks.
  5. **Repeat** of chunk 0 into a second cache, compared bitwise.
  6. **Peak GPU and host memory**, and images per second for each chunk.

  Task 0 also runs the LUMI interface check (timm PE support, cached weights and configs, including DINOv3-H+).
- New helpers: `scripts/compare_feature_caches.py` (repeat comparison), mean error and peak memory in `compare_with_v1_cache.py` and `extract_features.py`, and cached-config reading in `check_encoder_support.py`.
- **Local check** on the Dell (CPU, real DINOv3-L weights, the first 64 real benchmark images, chunks of 32):
  - extraction gave 2 chunks of 1024-d;
  - verification **PASS** (64 of 64 rows);
  - the rerun skipped 2 of 2 chunks;
  - the repeat was **bitwise identical** (max diff 0.0).

  The CPU speed (about 2.5 images/s) is no guide to GPU throughput.

**Waiting on LUMI:** syncing the code (git archive of the commit) and the manifest to `/flash/project_462001534/feature_geometry/manifests/`, then submitting.

**Estimate format once measured:** for each encoder, the images per GCD per second, dimension, disk (2,708,023 × dim × 4 bytes) and GPU-hours, kept separate from extrapolation, plus the proposed full-extraction job.

## 4b. Recommendations for the six candidates (`task.md` 8ac12a7)

Details and evidence: [docs/encoder_feasibility_audit.md](docs/encoder_feasibility_audit.md) section 5. No adapter is written.

| Candidate | Recommendation | Feature, dim |
|---|---|---|
| SigLIP 2 So400M | `google/siglip2-so400m-patch16-384` (`dd658faa`). The patch-16 series matches the other patch-16 models and has fewer tokens. **But** an official `patch16-256` exists, and the 256-preference rule would pick it: author to confirm 384 or 256 | attention-pooled, 1152 |
| SigLIP 2 Giant | `google/siglip2-giant-opt-patch16-256` (`46d7129e`), following the policy | attention-pooled, 1536 |
| PE-Core-G | Meta's timm port (`f537e434`), adopted only after a numerical check against the official code; otherwise the official code | image embedding, 1280 |
| C-RADIOv4-H | official implementation (`0057b339`, its remote code reviewed and pinned) at its preferred 512 px, with a 512 centre crop for batching. The crop affects only the 1.04 % of frames that aren't square | `summary`, dim to measure |
| AIMv2 | `apple/aimv2-large-patch14-224` (`fcb5093a`): same capacity and input size as the baselines, so this slot isolates the objective. Feature: mean of the final patch tokens (no AIMv2 checkpoint has a pooled output, `use_head: false`) | 1024 |
| DINOv3-H+ | `c807c9ee`, `pooler_output` | dim and cached weights **not yet confirmed** (LUMI blocked) |

## 5. Open decisions

1. **LUMI access:** answered. Logins are closed for a maintenance break (section 4). Nothing is needed from the author. The smoke test is submitted, within the approved scope, once a login succeeds.
2. **The six candidate recommendations** (section 4b), in particular SigLIP 2 So400M at 384 or at 256.
3. **Full extraction:** only after the smoke estimate, as `task.md` requires.
4. **Still open from before:**
   - aligning the README's synthetic-experiment wording with the rule;
   - the few-shot specimen-count mismatch.

**Stopped.**
