# Staged LUMI plan — NOT submitted (needs the author's approval; task.md 8538df5 Stage C)

Estimates from `readiness_summary.json` / `readiness_matrix.csv` (`scripts/benchmark/readiness_matrix.py`):
throughput 682.6 images/s per node, measured for DINOv3-L in job 23642060. Equal speed is **assumed** for DINOv2-L and
BioCLIP 2, and **1.4× slower** for SigLIP 2; these are extrapolations, to be replaced by the first job's measurement.
GPU-hours = node-hours × 4. Project 462001534. Paths: `F=/flash/project_462001534/feature_geometry`,
`S=/scratch/project_462001534/feature_geometry`, `C=$S/code/<commit>`.

## C0 — preparation (CPU and downloads; no GPU)

| Step | What | Command / script | Size |
|---|---|---|---|
| C0a | Pack Food-101, Pet, Aircraft into byte-verified HDF5 on /flash (content tables checked) | `pack_archive_hdf5.py --manifest <ds>.parquet --archives $S/benchmark_sources/<ds>/<archive> --content $S/benchmark_sources/<ds>/content.tsv --out $F/<ds>/<ds>_images.h5 --out-manifest $F/<ds>/<ds>_extract.parquet` (small partition, 1 task per dataset) | archives (`archives.tsv`): Food-101 4,996,278,331, Aircraft 2,753,340,328, Pet 791,918,971 bytes |
| C0b | Pack the BIOSCAN-5M closed-set images (from the cropped_256 zips) | same script, `--archives /scratch/project_462001534/aqua_v1/bioscan/zips` | 343,333 images; zips 2,224,548,648 + 1,467,020,347 bytes (`bioscan_zips_listing.txt`) |
| C0c | SigLIP 2 weights into the shared hf_cache at revision e8708ab7 | `huggingface-cli download google/siglip2-so400m-patch16-256 --revision e8708ab72d125807e45b36fb7d4e0aacbb59f379` (HF_HOME = the aqua_v1 hf_cache) | 4,542,792,928 bytes (`encoders/siglip2.json`) |
| C0d | BIOSCAN v1-cache equivalence: re-encode 512 closed-set images per encoder with this repo's extractor, compare with the v1 caches (`compare_caches_by_name.py`) | 3 short jobs (1 GCD each) | < 1 GPU-h |
| C0e | **DF24 images: needs a decision first.** Resolution (300 px, 6.42 GB, vs full size, about 110 GB) and the licence conflict (metadata "Public Domain" vs README CC BY-NC-ND 4.0) | download job after approval | 6.42 GB or ~110 GB (v1 dataset audit) |

## C1 — existing checkpoints (DINOv2-L, BioCLIP 2; DINOv3-L where not cached)

One `extract_encoder.sh` job per (encoder, dataset) on standard-g, 1 node:
`sbatch extract_encoder.sh $C <encoder> <extract manifest> --hdf5 <file.h5> <out>`. For CIFAR-100, use
`--cifar-dir <unpacked cifar-100-python>` with the smoke-tested path.

| Dataset | Images | DINOv2-L | BioCLIP 2 | DINOv3-L |
|---|---|---|---|---|
| FIN-Benthic2 (stored selection, `$F/finbenthic2/full_selection.parquet` + HDF5) | 186,284 | run | run | cached |
| AquaMonitor (stored selection) | 836,159 | run | run | cached |
| BIOSCAN-5M closed set | 343,333 | reuse v1 cache if C0d passes, else run | same | same |
| FGVC-Aircraft | 10,000 | run | run | run |
| Food-101 | 101,000 | run | run | run |
| CIFAR-100 | 60,000 | run | run | run |
| Oxford-IIIT Pet | 7,349 | run | run | run |

Estimated (if BIOSCAN is re-encoded and DF24 included): DINOv2-L 2.995 GPU-h, BioCLIP 2 2.995, DINOv3-L 1.331.

## C2 — SigLIP 2 (after C0c), all available datasets: estimated 4.194 GPU-h (incl. DF24)

## C3 — DF24 × 4 encoders (after C0e)

**Total, all 32 combinations:** about 11.515 GPU-hours (estimate) and 29.205 GB of float32 features. Every
extraction job ends with `verify_feature_cache.py` and a 256-image CPU-vs-GPU comparison (inside the job). Each new
encoder × dataset then gets a LayerNorm / rank audit on the train features before any native-space use.
