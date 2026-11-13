# Evidence — train+val frozen-feature extraction, 8 datasets × 4 encoders (task.md d57a630, C2 + C3)

**Result: 32 of 32 combinations have complete, checked train+val feature caches** (`completion_matrix.md`,
`completion_summary.json`). No test image was encoded for any new combination; nothing was scored.

## Files

- **`../c2_plan/`** — the plan, made before submission from the measured C1 rates (`scripts/benchmark/c2_plan.py`):
  - `plan_all.csv`: status per combination (24 new, 4 reused, 4 from C1), images, estimated encode time, time
    limit, storage, evidence for each reuse;
  - `plan.tsv`: the 24 submitted jobs (`scripts/lumi/submit_c2.sh`);
  - `records.tsv`: the inputs of the 32 C3 records (`scripts/lumi/c3_records.sh`);
  - `plan.json`: estimate of 11.2 GPU-h for the new extractions.
- **`extraction_summary_c2.{csv,json}`** — one row per new C2 cache (`scripts/benchmark/summarise_extraction.py`):
  rows, chunks, dimension, dtype, finite, norm range, encode seconds, images/s per GCD, cache verification, 256-row
  CPU-vs-GPU difference, peak GPU memory and host RSS. The C1 Aircraft caches are in `../c1_pilots/`.
- **`records/<encoder>__<dataset>.json`** — the C3 record of every combination
  (`scripts/benchmark/combination_record.py`, LUMI array 23697582, code 937cd61):
  - identity: manifest and standardised-manifest sha256; image ids equal to the standardised manifest's available
    train+val ids, no test id (single-split datasets); for FIN-Benthic2 / AquaMonitor, every id in the standardised
    manifest and all specimens present;
  - independent units expected vs present;
  - shape, float32, all finite, norms, sha256 of the features in manifest order;
  - numerical rank of the covariance and, for DINOv2-L / DINOv3-L, the final-LayerNorm constraint;
  - the source caches and the settings they record.
- **`completion_matrix.{csv,md}`, `completion_summary.json`** — `scripts/benchmark/completion_matrix.py`.
- **`sacct_d57a630.txt`, `resources.{csv,json}`** — every job of this task (C0 download and packing, C1, C2, C3)
  and the totals per stage (`scripts/benchmark/sum_resources.py`). Allocation-based, not LUMI's billing database.
- **`storage_and_quota.txt`** — `du` of the output folders and the (cached) `lumi-allocations` line.

## Jobs (all on project_462001534)

- **Batch 1:** FIN-Benthic2, Food-101, Oxford-IIIT Pet, CIFAR-100; GPU jobs 23696935–23696964 with CPU checks
  (`check_extraction.sh`), code a6ff351.
- **Batch 2:** AquaMonitor, BIOSCAN-5M, DF24; GPU jobs 23697108–23697124 with CPU checks, code a6ff351. Submitted
  after batch 1 had been inspected.
- **Records:** array 23697582 (32 tasks), started after every batch-2 check had succeeded.

## Storage

New feature caches: `/scratch/project_462001534/feature_geometry/benchmark_c2/<dataset>/<encoder>/features_gpu/`.
The reused caches stay where they were (paths in `../c2_plan/plan_all.csv`, column `record_sources`).

## Notes

- **Reading the energy column:** the `energy` TRES that Slurm records for these jobs adds up to only about 0.2 kWh
  for about 3 node-hours of GPU jobs. That is implausibly low for MI250X nodes, so it is listed but is not an energy
  measurement.
- **Five-fold datasets:** FIN-Benthic2 and AquaMonitor have no single test partition. Every specimen is test in
  exactly one fold and train or val in the other four, so the train+val union over the folds is the whole stored
  selection. Their caches are fold-independent, as are the earlier DINOv3-L caches. Test isolation is applied per
  fold at analysis time (`src/statistics/heldout_density.load_fold` drops the fold's test rows before any feature
  lookup).
