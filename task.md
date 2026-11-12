# Research coding tasks — Feature-Geometry

## Project operating agreement (2026-11-06)

**This is the new primary repository for future feature-geometry and frozen-representation research:** `mskhan793/Feature-Geometry`, branch `main`. The previous `mskhan793/Lightning_Aqua_Vision_Language_v1` remains an immutable source/reference for selective reusable Lightning infrastructure; it is not the active task/response exchange location.

- **Human + research assistant:** study math, derive formulas, review papers, develop one falsifiable hypothesis at a time, propose bounded experiments, critically inspect results, and give the coding assistant one task at a time. We must distinguish planned work from measurements actually run.
- **Coding assistant:** read **root `task.md`**, implement only an explicit approved current task, and write a concise **root `response.md`** when done (link longer reports, code, evidence and commits from there). Never infer authorization from the queue or notebook notes.
- **Task lifecycle:** task authored/reviewed -> implementation -> response.md with status and evidence -> research review -> next explicit task. Each new assignment replaces the active task section while preserving Git history; do not silently erase the project's working rules.
- **Modularity:** preserve effective source Lightning Dataset/DataModule/LightningModule behaviors, train/validation/test hooks, split semantics, metric/logging and configuration standards as relevant. Small focused Python files, real imports/tests, no dead code, no blanket copy of historical code, no speculative abstractions.
- **Safeguards:** train/validation/test independence, specimen/observation grouping, seeds, original experimental provenance. No dataset downloads, computational experiments, full migrations, external/GPU runs or publication claims without explicit task approval.
- **Separate paused direction:** multi-dataset taxonomic transfer extension (DF24/iNaturalist/NAFlora) is paused by the author. Handoff in old repository `docs/vlm_prototype_research/DATASET_EXTENSION_PAUSED_HANDOFF_2026-11-06.md`. Do not auto-restart.
- **Migration status:** the source-to-new-repo clean Lightning transfer was requested conceptually. **It is not an instruction to bulk-migrate now.** First inventory the old code and propose a small source-to-destination map in a dedicated approved task; then review before code transfer. Preserve this repo's existing scaffold, if compatible, instead of overwriting it.

## Long-term research vision — geometry-informed model and method selection (2026-11-07)

**Central question:** *Can statistical and geometric properties of frozen pretrained visual representations predict which pretrained encoder and simple classifier will work well, and when the additional cost of full fine-tuning is justified?*

This is a **research hypothesis and program, not an established result or authorization to run experiments**. Study what is already known, then formulate falsifiable hypotheses and test them. The long-term aim is a rigorously evaluated, computationally efficient protocol for choosing pretrained models and downstream learning strategies for fine-grained and other image classification tasks.

**Three connected research goals:**

1. **Understand representation geometry.** Analyze class-mean separation, within- and between-class covariance, eigenspectra, effective rank, conditioning, class imbalance, and the effect of model pretraining objective, domain specialization, capacity and feature dimension. Investigate simple Euclidean/cosine NCM, linear probes, shrinkage/Mahalanobis classifiers, Gaussian density/GMM methods, and related transferability criteria (including H-score). Maintain a clear distinction between descriptive statistics and validated predictive indicators.
2. **Predict suitability and avoid unnecessary expensive adaptation.** Ask whether statistics measured from **official training embeddings** can forecast absolute frozen-feature performance, the *relative* performance of candidate classifier families, and the **incremental benefit** of fine-tuning. The last claim **requires actual, fairly budgeted fine-tuning baselines** for empirical validation; frozen embeddings or high H-score alone cannot prove fine-tuning is unnecessary.
3. **Develop and evaluate a model-selection protocol.** Rank encoder–classifier combinations using low-cost feature diagnostics. Compare its agreement with official held-out downstream outcomes against simple alternatives (e.g. NCM/linear-probe validation), measure ranking/regret/calibration where suitable, account for feature-extraction + selection + training compute, and test generalization to **independent benchmarks and visual domains**. Consider dataset/source-pretraining overlap, model families sharing teachers, input resolution, feature dimension, class prevalence and aggregation unit when interpreting evidence.

**Experimental integrity and sequencing:**
- The current approved deliverable is **only a high-integrity full-dimensional frozen-feature cache for AquaMonitor Classification**, across its five official folds and the encoder shortlist described below; model additions still require checkpoint/feasibility verification and any expensive run still requires explicit approval. **Do not turn this vision into an automatic experiment queue.**
- Future research tasks are chosen **one hypothesis and one method at a time** after reviewing literature and current measurements. Use official publisher-provided partitions only, never new scientific train/val/test partitions. Fit statistical procedures using official training data, select methods on official validation, and use test only for locked evaluation. Across-benchmark claims require independent dataset replication.
- Preserve **native/full-dimensional outputs**; study PCA, Gaussian random projections and normalization later as explicit, separately approved transformations, never as silent feature-cache preprocessing. Projection can alter discriminative information and geometry.
- Report negative results, uncertainty, dependence of observations within specimens/classes, computational overhead and failure cases. Do **not** assume novelty, that simple statistics outperform fine-tuning, or that fine-tuning can be predicted reliably in advance. Separate correlation from causal explanations.

**Literature and learning:** This program builds on the ongoing mathematical study of Mensink et al. (2013), Ibrahim et al. (2023), and Cao et al. (ICML 2026), among other work. Keep the separate `Feature_Literature` slides repository for theory notes; `Feature-Geometry` remains the implementation/results repository. The third paper is still in progress.

## Standing dataset and split rule — AUTHOR DECISION (2026-11-06)

**Default: every research experiment uses a real established benchmark and published train/validation/test partitions where available.** The author has expressly approved two *named, reproducible exceptions* on 2026-11-10: FIN-Benthic2's existing independently derived, specimen-disjoint five-split CSV; and a newly derived DF24 validation subset drawn solely from the authors' official training observations. These are **not official split releases** and must always be labeled as derived. The original published FIN-Benthic2 splits remain available as an optional later sensitivity comparison, not this chosen protocol. No synthetic research datasets, Gaussian toy populations or generated feature datasets. No ad-hoc partitions other than the two specifically approved and documented exceptions; do not invent or regenerate partitions independently at each run. Do not quietly replace a missing official validation/test split with a new split. If a benchmark lacks a suitable official partition, report that limitation and propose another benchmark or explicitly request an exception from the author **before** proceeding.

- Select the benchmark, its exact version, published split/protocol and evaluation unit before implementation. Obtain explicit approval for dataset downloads and compute.
- Preserve original official partitions and all IDs; avoid moving examples between splits or merging train with val/test. Apply any exclusion of target species from metric learning *within the training partition only*, if that research protocol is separately approved, without inventing new sample partitions.
- Compute statistics/fit models on the designated **training partition** (official or explicitly approved derived protocol) only; select hyperparameters on its corresponding **validation partition** (official or explicitly approved derived protocol) only; use its untouched **test partition** only for final locked evaluation. For DF24, the test is the authors' original `pubtest`. Respect grouped observations/specimens and document any official-protocol limitation or leakage risk.
- Frozen features extracted from these original images inherit the **same** source labels, partitions and sample identifiers.
- Small mocked tensors/fixtures are permitted **only for unit/software correctness tests**, not as research datasets, scientific comparisons or reported experimental evidence. This exception does not permit synthetic benchmark partitions.
- This rule supersedes all earlier proposed synthetic Gaussian experiments in this task file or README; the existing exploratory code is not approved for experimental use on its own.

## Current state
Repository scaffold initialized. No experimental claims or benchmarks have been run.

## Working rules
- Continue one mathematical idea at a time alongside our paper-study discussion.
- Do **not** implement all experiments at once. Each implementation task must be explicitly scoped and accepted independently.
- Prioritize mathematical correctness and reproducible evaluation on official real-benchmark partitions. Use small code-level fixtures only for software tests.
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
5. **Split discipline:** fitting on train only, tuning/model selection on validation, final measurement on held-out test. No research split derivation except the two named author-approved, recorded exceptions in the standing policy. Preserve each benchmark's official partitions and specimen/observation-level grouping when relevant.
6. **Reproducibility:** explicit configs, seeds, dataset versions, normalization, estimator regularization, saved predictions and metrics, checkpointing only where training applies, unit and data-module integration tests.
7. **Scope discipline:** this chat discusses mathematical theory and interprets results; separate coding sessions act on the single explicit task written below. Before implementation, discuss and approve that task's scope; never silently implement the entire queue.

## Current active task — Standardize the eight-dataset, multi-foundation-model benchmark (2026-11-12)

**Author decision:** Move quickly to a broad generalization benchmark and **pause further taxonomy-based covariance pooling**. The previous class-balanced vs taxonomic/similarity pooling experiments are **completed** (see Git history/previous `response.md`), not deleted. Class-balanced pooling may remain a labelled experimental reference, not a permanent method or automatic default. This task is **benchmark standardization + feasible data/encoder onboarding and a bounded extraction smoke test**, NOT an uncontrolled full 8 × 4 extraction/methods sweep. Aim to produce a reproducible, comparable, execution-ready standard and a concrete LUMI compute plan promptly. Do not ask for further design preferences already specified below; explicitly ask only about genuine missing licenses/access, unavailable data/papers or large GPU-budget approval.

### 1. Fixed eight datasets — all must be accounted for

| Group | Dataset | Splits / statistical unit / constraints |
|---|---|---|
| Biology | **FIN-Benthic2** | Preserve user's original reproducible five specimen-disjoint folds (train/val/test); individual specimen is independent unit; paired cameras/frames aggregated identically to prior work |
| Biology | **AquaMonitor Classification** | Preserve original published five folds; specimen unit, camera-balanced aggregation, train/val/test locked; retain source-dataset metadata as descriptive diagnostic |
| Biology | **BIOSCAN-5M** | Use the existing verified official species train/validation/test metadata; audit actual image/specimen identity, group repeated records when possible, long-tail support/eligibility and available images; never promote val/test into train |
| Biology | **DanishFungi24 (DF24)** | Official train/public-test; preserve the author's already approved **derived grouped validation** from official train (seed 20261110) and grouping by observation; verify exact downloaded-image availability and leakage checks |
| General | **FGVC-Aircraft** | Verify original fine-grained variant release, publisher's train/val/test files and versions; use official splits and variant label granularity consistently; image is unit unless verified repeat IDs exist |
| General | **Food-101** | Original publisher train/test; create one fixed, stratified **train-only** validation with recorded seed and class counts, never touch official test |
| General | **CIFAR-100** | Original publisher train/test; create one fixed, stratified **train-only** validation; record 32×32 image resolution, preprocessing/upscaling decisions and 20 coarse categories; image unit |
| General | **Oxford-IIIT Pet** | Original publisher trainval/test; derive fixed stratified validation from **trainval only** (no leakage to test); label = 37 breeds; validate duplicate/grouping constraints |

**Do not reduce to a subset merely because the preparation is difficult.** If a dataset is blocked, label it BLOCKED with explicit reason/action and finish the readiness report for all others; do not fabricate missing source IDs, images, taxonomy, official folds or licenses. Keep published test data locked until explicitly authorized for a final evaluation; run initial research fitting on train and measurements on validation only.

### 2. Foundation-model candidates (frozen visual encoders)

- **DINOv3-L** (existing, verified checkpoint/preprocessing/cached embeddings for the two aquatic datasets).
- **DINOv2-L** (general self-supervised comparator).
- **BioCLIP 2** (biology-specialized image-text pretrained visual tower).
- **SigLIP 2** (general image-text pretrained visual tower).

For each: verify exact original publisher checkpoint and revision, architecture/embedding dimension, extraction layer/token/pooling, image normalization, input resolution, preprocessing, usage/license/access requirements, and final feature L2 normalization status. Do not assume that model marketing names are unambiguous; report checkpoint IDs before bulk use. For each dataset, use *the same eligible image identities across models* where source data permits and compare train/val/test identities exactly. Keep native features unnormalized and any L2-normalized alternative as separate named arm; never silently alter the original DINOv3-L cache convention. Record and audit any LayerNorm rank constraint (known existing DINOv3-L last-norm outputs: 1024 stored but 1023 linearly independent), numerical rank, and zero-eigenvalue handling per model; do not assume all backbones share the same constraint. No fine-tuning or learned head in this task.

### 3. Mandatory common benchmark standard

1. **Original provenance:** Locate and read original publisher dataset docs/data papers and any checkpoint model cards; inspect release versions, licensing/redistribution, split manifests and checksums. Download and store readable local source papers/docs **before relying on their detailed claims**; verify official author code when a published method is run. If essential materials are unavailable and cannot be downloaded, explicitly ask for the original PDF, data or access rather than inventing substitutes. Where a source has no downloadable paper, use and cite its official documentation and mark absence.
2. **Immutable standardized manifests:** Per dataset create deterministic table of dataset/version, source ID, specimen/observation grouping key when available, class label, original split, derived split flag/seed, repeat/near-duplicate warnings, image path/URL, exact file availability, and taxonomy or coarse labels when published. Hash source and output manifests. Preserve all previously frozen split files unchanged. No synthetic specimens or label invention.
3. **Evaluation units:** FIN-Benthic2/AquaMonitor = independent specimen, with prior camera-balanced mean-of-selected-frames protocol (up to 10 frames per camera; preserve stored selection). DF24 = observation/group (multiple photos of one observation treated together), BIOSCAN = confirmed independent specimen/image key (audit first), FGVC-Aircraft/Food-101/CIFAR-100/Pet = image unless verified duplicate/group IDs warrant group isolation. Never count repeated views as independent for covariance, bootstrap or support thresholds.
4. **Class support, fairness and imbalance:** Report full data size and class count, per-class train/val/test support, rare-class coverage and long-tail percentiles. Define a transparent *common, feasible* core eligible set for Gaussian full-covariance analysis (e.g. ≥15 train and ≥5 val independent units, as earlier) and always disclose excluded class counts; publish a separate all-classes / rare-class descriptive track when estimation is infeasible. Compare methods/models on **paired matching class sets and identical specimen/image IDs**, not only the overlapping reliable predictions chosen post hoc. Do not artificially balance evaluation by dropping real val samples; class-macro and micro are both reported.
5. **Feature-space convention:** Frozen visual representations, extraction/reproducibility logs, caching, numerical validity, train-only PCA (basis trained separately per dataset × model × split) and no validation/test fitting. For geometry/density work, specify **PCA-16 and PCA-64** primary comparable dimensions; if a model's rank or sample support prevents a dimension, declare unavailable rather than padding. Keep paper-faithful **Pándy spherical PCA-64 GBC** as an anchored baseline; **per-class LW full covariance** and covariance-condition diagnostics as separate arms; report projection variance explained and class support. Do not combine distinct density metrics or dimensions into one misleading rank. FDA and author-paper random projection (Ibrahim et al., q=128) are **later, separately scoped** comparisons after data/encoder baseline readiness; random seeds must be repeated and held fixed across comparisons.
6. **Metrics and uncertainty:** Initial descriptive primary: specimen/observation-level held-out class-conditional Gaussian log-likelihood (class-macro, micro secondary), native/PCA covariance spectra and conditioning, (d′) defined along the train-centroid direction and measured on val, and pairwise Bhattacharyya overlap with clearly stated covariance form, dimensions and eligible class pairs. Do not interpret likelihood or Gaussian class overlap as proof of recognition quality. Later, when explicitly approved, evaluate a **small fixed classifier suite** (e.g. cosine/Euclidean nearest-class-mean and train-only linear probe) and macro-F1/balanced accuracy on locked evaluation to test metric predictiveness. Do **not** train or evaluate classifiers now. Conditional confidence intervals via independent-unit bootstraps, folds reported individually with overlap of folds acknowledged; seeds and exact support denominators.
7. **Compute/organization:** reuse existing DINOv3-L aquatic embedding caches; *never* re-extract those without demonstrated mismatch. Separate `src/data` dataset adapters and manifest verification, `src/encoders` checkpoint-specific preprocessing/extraction, `src/statistics` shared metrics, `scripts/lumi` bounded jobs. Modular tests for split leakage, group units, source checksums, shape/dtype, model-specific preprocessing, rank, and paired class sets. Store large images/embeddings outside Git, tracked hashes+manifests/evidence in Git.

### 4. Fast staged execution / precise scope now

**Stage A — DO NOW (all eight datasets and four model cards):**
- Audit what is already physically available on Dell/LUMI vs only metadata; locate verified original download sources and official split files for all eight. Avoid bulk image downloads until licensing, disk and bandwidth check.
- Build or adapt versioned dataset manifest generators and **immutable** split summaries, with exact train/val/test/independent-unit counts and per-class support. Retain existing four biological split decisions without modification. For Food-101, CIFAR-100 and Oxford-IIIT Pet, create deterministic validation manifests from the appropriate source-train partition and document the split proportion/seed. For Aircraft, verify official val is actually provided and exact version.
- Verify four exact frozen encoder checkpoint IDs and legal access, preprocessing/feature-layer choices and expected GPU/CPU/storage cost. Build minimal adapter interfaces/smoke tests **without downloading multi-GB weights solely to prove API shape** unless readily available.
- Produce one concise 8 × 4 readiness matrix (32 combinations: ready/missing images/missing checkpoint/blocked), storage/throughput estimate and prioritized LUMI batch plan. Prioritize existing DINOv3-L aquatic baseline, then one biological + one general-image pilot for new encoders, then broader rollout; every one of the eight remains in scope for preparation.

**Stage B — LIMITED SMOKE TEST ONLY, if local data and existing checkpoints permit without large new downloads:** extract a small fixed number of actual training images/units for one new encoder and one new dataset, check reproducibility, preprocessing and specimen grouping; do **not** use val/test scores for choosing encoders and do not claim the full benchmark is complete. Report sample IDs and whether embeddings reproduce consistently.

**Stage C — NOT AUTHORIZED YET:** expensive bulk image downloads, full 8 × 4 GPU extraction, full repeated-seed experiments, tuning and test-set evaluation. Present a bounded resource estimate and explicit proposed batches for author approval in `response.md`.

### Deliverables / stop point

- `docs/benchmark_standard.md` specifying every dataset's exact official/derived splits, independent-unit convention, model-specific feature protocol, primary/secondary metrics, class eligibility, no-leakage guarantees, and what is/not comparable across encoders and datasets.
- `docs/evidence/benchmark_readiness/`: 8 dataset-source audits, manifests/source hashes and class-count diagnostics, all four encoder audit records/checkpoint IDs, 8 × 4 readiness matrix, pilot results if feasible, estimated storage/GPU/CPU costs and a runnable but **not automatically submitted** staged LUMI plan.
- Root `response.md`: short readiness summary, exact completed files/actions, blockers and license issues, baseline reuse, exact approval needed to execute bulk extraction; no invented outcomes.
- **STOP for review** after Stage A and any truly small Stage B. No taxonomy-informed pooling, no methods sweeps, classifier training, broad extraction, or new research findings claimed without actual experiments.

**Priority:** This supersedes the *completed* class-balanced/taxonomic covariance experiment as the only active assignment. The taxonomy branch is paused; do not resume it as part of the benchmark. Other queued items below remain inactive.

## Current approved task — Prepare AquaMonitor Classification frozen features

**Author decision, 2026-11-06:** Use the original **AquaMonitor Classification** benchmark for our first real-data work. Cache features for **all official benchmark images**, reusable across **all five original folds**. No classification/statistical experiments yet. Later explore additional established classification benchmarks, each with its own published partitions; do not automatically resume the paused DF24/iNaturalist/NAFlora taxonomic-exclusion project.

### Official checkpoint resolution policy (author decision)

**We won't arbitrarily resize models to 256. We'll select the appropriate official checkpoint and its corresponding preprocessing.**

When a model family has multiple **official, pretrained checkpoint variants** at different input resolutions, prefer the **official 256-pixel variant over the official 384-pixel variant** when suitable and available. This does **not** require every encoder to use 256 pixels: retain native supported resolution/preprocessing where there is no suitable 256 variant (e.g. existing BioCLIP-2 and DINOv2-L 224-pixel pipelines), and do not alter the existing v1 equivalence preprocessing for the first three baselines.

Document exact checkpoint revision, feature output, input resolution and published preprocessing for each encoder. A 256-pixel checkpoint and a 384-pixel checkpoint are distinct model configurations; do not simulate one by simply resizing inputs to the other. Treat within-model resolution sensitivity as an optional separate future experiment, not this extraction task.

### Immediate authorized checkpoint — three-model LUMI smoke test and variant decisions (2026-11-07)

**Authorization:** Proceed with the coding assistant's proposed **small LUMI `dev-g` smoke test only** for the three already implemented frozen encoders: **BioCLIP-2, DINOv2-L, DINOv3-L**. Maximum requested scope: **one node, up to one hour / approximately four GPU-hours total** on project `462001534`. Stop if resource constraints would exceed this cap and report instead of escalating the run. This is an infrastructure check, **not** an experiment, model selection or full-feature extraction.

1. Sync only the necessary new-repository code and the verified classification manifest to LUMI; use existing verified image HDF5, cached official weights, container and binding setup; do not recopy/download full source images or checkpoints unless explicitly justified and separately approved. Confirm LUMI connectivity, paths, container bindings and access before submitting.
2. For **each of three** encoders, reproduce the v1 Aqua frozen-feature output on the agreed fixed **512 real images** with the exact matching preprocessing/weights (report max/mean error, tolerance and pass/fail); run a short deterministic real-image extraction subset from official Classification images with ID/shape/finite checks; verify chunk restart/resume and repeatability; measure actual throughput and peak memory. Smoke-subset selection **does not** create a new research partition, and its outputs are not classification evidence.
3. Record job ID, GPU allocation/elapsed usage, software/environment versions, transforms, checkpoint identifiers and immutable revisions, input/image identity, feature dimensions, cache path/checksums and any failures. Make per-encoder estimates (plus total) for storing and extracting **2,708,023 unique images**, explicitly separating measured numbers from extrapolation, and include concrete proposed full-extraction command/Slurm plan. Report estimates before proposing full runs.
4. **Do not launch any full dataset extraction or any accuracy, statistical, classifier or fine-tuning experiment.** Return results to root `response.md` and stop for author review. If LUMI is unreachable, state blocker; do not fabricate smoke/throughput results.

**Six additional model candidates:** The metadata audit in `docs/encoder_feasibility_audit.md` is complete, but model variants have not been approved. In the **same response**, recommend one **specific verified checkpoint and feature definition** for each unresolved case with evidence and rationale:
- SigLIP 2 So400M: patch-16 vs patch-14, held at 384 px;
- SigLIP 2 Giant: 384 vs 256 input resolution;
- PE-Core-G: original official implementation vs documented timm port;
- C-RADIOv4-H: trusted official implementation, input resolution and representation to retain;
- AIMv2: specific pretrained variant, resolution and reproducible patch-token pooling;
- DINOv3-H+: confirm cached gated weights, feature interface and output dimension on LUMI.

Evaluate choices for scientific comparability (native feature fidelity, consistent model inference, accessibility, licenses, computational cost and reproducibility), **not just convenience**. Include exact revision identifiers, feature dimension and preprocessing, and identify any uncertain/unavailable candidates. **No new adapters or full extraction of these six are authorized by this checkpoint** until the author reviews model configurations. The planned nine-model lineup is an exploratory comparison, not nine independent pretraining samples.

### Data and split contract
- Canonical local metadata: `~/Documents/Research/Feature_geometry_data/Aqua_Paraquete_files/aquamonitor-classif.parquet.gzip`; use the original authors' parquet, **not** the derived `*-with-taxonomy_all*` file.
- **Classification only:** 42 closed-set classes, label column `taxon_group`, official `fold0` through `fold4`. The source inspection recorded **2,708,023** eligible image rows (2,756,664 total, 48,641 marked outside the benchmark as `'0'`); independently recheck before running. For fold0: 1,882,046 train, 282,923 validation, 543,054 test images; 29,346 / 4,386 / 8,433 specimens. Do **not** generate new splits or move specimens.
- Preserve stable image ID/source row, `individual` specimen ID, `taxon_group`, all five split labels, and sufficient metadata for downstream grouping. Encode a unique image **once**, join cached features to all folds later. Do not reduce a specimen's multiple frames to an average during extraction; image-level raw features are the primary cache.
- Confirm where original image bytes are located in local/LUMI storage; parquet may contain metadata only. Do not download large data without author approval.

### Implement minimal extraction infrastructure
1. Inspect and selectively reuse existing `Lightning_Aqua_Vision_Language_v1` data/image pipelines where compatible, preserving meaningful `torch.utils.data.Dataset`, `LightningDataModule` and predict lifecycle behavior. Small, focused files only; no blanket copying, dead code or complex abstractions.
2. Build an image manifest and read-only integrity audit of all five original folds. Report missing/duplicate identifiers, label inconsistencies, class counts, specimen disjointness and image availability.
3. Preserve the **three implemented baseline extractors** (BioCLIP-2, DINOv2-L, DINOv3-L) and **extend the planned encoder portfolio** to the following nine distinct checkpoint configurations. The additional six are author-discussed research candidates; their **metadata feasibility audit is complete** (see `docs/encoder_feasibility_audit.md`), but variant selection, LUMI compatibility and implementation remain pending:

   | Role | Encoder configuration | Status |
   |---|---|---|
   | Biology-specific vision–language | BioCLIP-2 | Implemented; needs real-image equivalence check |
   | Image-only self-supervised | DINOv2-L | Implemented; needs real-image equivalence check |
   | Image-only self-supervised | DINOv3-L | Implemented; needs real-image equivalence check |
   | Capacity scaling | DINOv3-H+ | New candidate |
   | General image–text alignment | SigLIP 2 So400M (384-pixel variant) | New candidate |
   | Large image–text scaling | SigLIP 2 Giant | New candidate; resolve exact checkpoint/size |
   | Contrastive perception encoder | Meta PE-Core-G | New candidate; resolve exact released checkpoint |
   | Multi-teacher representation | NVIDIA C-RADIOv4-H | New candidate; resolve exact released checkpoint |
   | Multimodal autoregressive | Apple AIMv2 | New candidate; resolve exact variant and feature interface |

   **First do a metadata/checkpoint feasibility audit for the six new candidates:** source publication and release, current access/permissions/license, exact model ID and immutable revision, parameter count, feature dimension, whether outputs are native or normalized, image input resolution/eval preprocessing, extraction interface, local/LUMI runtime compatibility, and whether external access or gated weights require author action. Cite authoritative model documentation in an evidence report and distinguish verified fields from unknowns. **Do not automatically substitute a similarly named model.** Flag any unavailable candidate and recommend a scientifically comparable replacement for author review. Since model objectives/pretraining corpora differ, record known potential source-data overlap rather than assuming model independence.
   
   Implement the six additional adapters **only after the author approves their verified model variants in a subsequent task**, with a minimal real-image smoke check each; do not rewrite or disturb the baseline extractors. No full extraction for any encoder without the compute authorization gate below. Use deterministic eval-mode inference with no gradients or stochastic augmentation.
4. Cache **raw/native model-output embeddings** as the canonical representation, describing any built-in normalization; if unit-normalized embeddings are later needed, derive them transparently, not as a silent replacement. Track image ID, label, specimen, original fold memberships, encoder/checkpoint/preprocess version, feature dimension, dtype, norms and checksums. Store in chunked/resumable files outside Git under `Feature_geometry_data` or approved LUMI storage; guard against duplicates and partial writes.
5. Use a real Lightning DataModule and `LightningModule.predict_step`/`Trainer.predict` where sensible. Stream batches and avoid holding all images/features in memory. Make a deterministic **small subset of actual official images** a software smoke test only, never a new research partition.
6. Test ID ↔ feature correspondence, repeated inference consistency, resume behavior, fold preservation, finite outputs, and known source counts. Add concise README setup and extraction instructions. Align README's obsolete synthetic-experiment wording with the official-benchmark-only rule; preserve reusable math utilities unless there is a justified reason to remove them.

### Mandatory computational gate
- **Approved now:** the specific three-baseline LUMI smoke test defined above (subject to the stated small compute cap), metadata/code checks, feasibility recommendations and a grounded compute/storage estimate. No other GPU tasks are implied.
- **Do not automatically start** extraction over ~2.7 million images × up to nine encoders, download huge archives or submit large LUMI/GPU jobs. Present per-encoder exact images/dimensions for all verified candidates, disk estimates, throughput measured on a small batch if possible, runtime and billing estimates with assumptions, and the proposed extraction command/job. Obtain author approval for full GPU execution first.
- After explicit approval, extract all benchmark image features and verify complete coverage; do not silently skip corrupt images. Stop and report in root `response.md` with linked evidence and unresolved exceptions.
- **Not authorized:** NCM, linear probe, covariance/GMM comparisons, learned W, classification evaluation, hyperparameter tuning, new dataset experiments or any final test scoring.

### Response expectations
Record source dataset version, nine-candidate verification matrix (with missing/unavailable checkpoints explicitly marked), checkpoint IDs for verified models, source-to-new-code migration map, created files, counts for the five official folds, tests genuinely executed, exact output paths and hashes, workload estimate, and questions blocking full extraction. Report progress in **root `response.md`**, then stop for our scientific review.

## Task queue (planned, not started)
1. **NCM / metric geometry:** Compare Euclidean and cosine NCM on an approved benchmark's official splits; analyze observed class separation and within-class variation. Later evaluate learned/shared Mahalanobis variants.
2. **H-score / shrinkage:** Implement original and shrinkage H-scores; inspect covariance conditioning, Ledoit–Wolf estimation, and common-dimensional random projection on real training embeddings; validation for method selection.
3. **Gaussian feature replay:** If explicitly approved, evaluate class-wise Gaussian feature replay using distributions estimated **only from official training embeddings**, never as a substitute for real benchmark evaluation; compare against actual official validation/test images. Generated features must never define test partitions.
4. **Gaussian mixtures:** Fit models to actual frozen benchmark training embeddings; compare prototype, single Gaussian and GMM inference on official validation/test partitions.

## Definition of done for each task
- Minimal tested code in `src/` when appropriate
- Runnable script/notebook with deterministic seed
- Exact real benchmark release, official split identifiers, leakage checks and evaluation metric
- Short results file with what worked, what failed, and limitations
- No hidden modifications to unrelated experiments
