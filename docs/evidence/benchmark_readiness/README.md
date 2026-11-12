# Evidence — benchmark readiness (task.md 8538df5, Stage A + Stage B smoke test)

- **`datasets/<dataset>.json`** — one per dataset (8). Each records:
  - source files with sha256 and the manifest sha256;
  - images, units and classes per split (or per fold);
  - eligibility (≥ 15 train, ≥ 5 val units) and long-tail percentiles;
  - units in several splits, units with several labels, images sharing a unit.
  - Built by `scripts/benchmark/build_manifests.py`. The manifests themselves (parquet) are in
    `~/Documents/Research/Feature_geometry_data/benchmark/manifests/` (outside Git).
- **`encoders/<encoder>.json`** — one per encoder (4): Hugging Face revision, licence / gating, parameters, weight
  files, config and processor files, feature definition, L2 / LayerNorm notes (`scripts/benchmark/audit_encoders.py`).
- **`readiness_matrix.{csv,md}`, `readiness_summary.json`** — the 8 × 4 status, images to encode, time and storage
  estimates (`scripts/benchmark/readiness_matrix.py`). `throughput_reference_23642060.txt` holds the measured
  throughput job's accounting.
- **`lumi_plan.md`** — staged, runnable, not-submitted plan.
- **`smoke_dinov2_l_cifar100/`** — Stage B results:
  - LUMI job 23687426 (dev-g, 1 GCD, 1:18), code 75e8c7c;
  - 512 fixed CIFAR-100 training images (`sample_ids.json`; seed 20261112);
  - two extractions, bitwise identical (`repeat_comparison.json`);
  - LayerNorm constraint audit (`null_space_dinov2_l_cifar100.json`);
  - logs.
- **Sources** (outside Git):
  - papers and official pages in `~/Documents/Research/methods/benchmark_sources/` (`sha256.txt`; fetched by
    `scripts/benchmark/fetch_sources.sh`);
  - archives on LUMI in `/scratch/project_462001534/feature_geometry/benchmark_sources/`: downloaded by job 23686949
    (which stopped at the Aircraft MD5 check after all downloads had finished), then verified by job 23687216:
    `archives.tsv` with sizes, MD5, sha256, gzip test; content tables; split files.
  - Local copies of the split files and content tables: `~/Documents/Research/Feature_geometry_data/benchmark/sources/`.
