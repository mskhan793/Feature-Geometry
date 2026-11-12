# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 8538df5, "Standardize the eight-dataset, multi-foundation-model benchmark".
**Status: Stage A done for all 8 datasets × 4 encoders (32 combinations); one Stage B smoke test done; stopped for
review.**
- **Not started (Stage C, needs approval):** no bulk extraction, no experiments, no classifiers, no test evaluation.
- **Taxonomy branch:** paused.
- **Standard:** `docs/benchmark_standard.md`. **Evidence:** `docs/evidence/benchmark_readiness/` (README).

Previous response (class-balanced / taxonomic pooling): commit 0a9d7ba.

## 1. Readiness (8 × 4)

| Dataset | DINOv3-L | DINOv2-L | BioCLIP 2 | SigLIP 2 |
|---|---|---|---|---|
| FIN-Benthic2 | cached, verified | ready | ready | missing checkpoint |
| AquaMonitor | cached, verified | ready | ready | missing checkpoint |
| BIOSCAN-5M | cached in v1 (equivalence check pending) | cached in v1 (pending) | cached in v1 (pending) | missing checkpoint |
| DanishFungi24 | missing images | missing images | missing images | missing images + checkpoint |
| FGVC-Aircraft | ready | ready | ready | missing checkpoint |
| Food-101 | ready | ready | ready | missing checkpoint |
| CIFAR-100 | ready | ready | ready | missing checkpoint |
| Oxford-IIIT Pet | ready | ready | ready | missing checkpoint |

- **"Ready"** = images and checkpoint on LUMI, manifest built.
- **"Missing checkpoint"** = the SigLIP 2 weights (`model.safetensors`, 4,542,792,928 bytes) are not yet downloaded. They are open (Apache-2.0); the
  adapter is written and tested on a tiny random model.
- **DF24 images:** not downloaded (§4).
- **No dataset is BLOCKED.**

## 2. What was done

- **Sources** (read before relying on them): papers and official pages for FGVC-Aircraft, Food-101, CIFAR and
  Oxford-IIIT Pet, plus the SigLIP 2 paper, saved with sha256 (`~/Documents/Research/methods/benchmark_sources/`).
  BIOSCAN-5M, DINOv2, DINOv3 and BioCLIP 2 papers were already local.
- **Archives on LUMI** (`$S/benchmark_sources/`; small CPU jobs 23686949, 23687216):
  - **CIFAR-100:** publisher MD5 eb9058c3 verified (Dell and LUMI).
  - **Aircraft:** the published MD5s **do not match** either the full or the annotations-only archive. Two
    independent downloads are byte-identical (sha256 e4e323d4), and their split files are identical to the
    annotations-only archive's. Most likely the published checksums are stale; it can't be confirmed against the
    publisher — a provenance caveat, not a blocker.
  - **Food-101 and Pet:** no published checksums; sha256 recorded.
  - Every archive passes `gzip -t`, and every image member is hashed.
- **Eight manifests** (`src/data/benchmark_manifests.py`, `scripts/benchmark/build_manifests.py`) in one schema, with
  source and manifest hashes and per-class support. The frozen FIN-Benthic2 folds, AquaMonitor folds, BIOSCAN-5M
  splits and DF24 derived validation are reproduced unchanged. New derived validation (10 % of the official train /
  trainval per class, seed 20261112, units whole): CIFAR-100, Food-101, Pet. Aircraft has an official val.
  - Counts: `datasets/*.json` and `docs/benchmark_standard.md` §1, §3.
  - **Findings:**
    - CIFAR-100's publisher file names are not unique (array position is used as the id);
    - CIFAR-100 has 26 exact-duplicate groups: 10 cross the official train / test boundary, and 17 carry
      conflicting labels;
    - Food-101 has 16 images sharing a unit; 2 cross train / test, 2 carry conflicting labels;
    - DF24's label is `scientificName` = `class_id` (1604): its `species` column merges 25 classes;
    - BIOSCAN-5M: 531 of 11846 species meet ≥ 15 train / ≥ 5 val.
- **Encoder audit** (`encoders/*.json`; metadata only): pinned revisions DINOv3-L ea8dc286, DINOv2-L 47b73eef,
  BioCLIP 2 2957b322, SigLIP 2 so400m-patch16-256 e8708ab7.
  - The 256 variant was chosen by the existing checkpoint policy; 384 is recorded as the alternative.
  - Revisions are now pinned in the encoder registry.
  - DINOv3's official processor default is 224, but the existing caches use 256 × 256; this is kept and documented.
- **Code:**
  - CIFAR pickle image source;
  - SigLIP 2 adapter (the checkpoint's own processor; pooled vision output);
  - generic archive → HDF5 packer (byte-verified);
  - encoder-agnostic extraction job (`extract_encoder.sh`; `extract_dinov3_l.sh` now delegates to it, same steps);
  - readiness matrix.
  - Tests: 105 pass.

## 3. Stage B smoke test (DINOv2-L × CIFAR-100; LUMI 23687426, dev-g 1 GCD, 1:18)

- **Sample:** 512 fixed training images (seed 20261112, all 100 classes; ids in `smoke_dinov2_l_cifar100/sample_ids.json`).
- **Reproducibility:** two independent extractions are **bitwise identical** (512 of 512).
- **Preprocessing:** decoding checked against the raw arrays; the 32 × 32 images are upscaled only by the encoder's
  own transform (Resize 256, CenterCrop 224).
- **LayerNorm constraint:** **confirmed for DINOv2-L** (relative violation ≤ 9.3e−7), as for DINOv3-L. Rank can't be
  measured from 512 < 1024 samples.
- **No val or test data used,** and no claim about encoder quality. The 20.7 images/s on one GCD is startup-dominated
  and not a throughput measurement.

## 4. Decisions / approvals needed (Stage C)

1. **Bulk extraction**, as in `lumi_plan.md`:
   - C0: pack 4 datasets to /flash, download SigLIP 2 weights, BIOSCAN v1-cache equivalence check;
   - C1 / C2: extraction jobs.
   - Estimate for all 32 combinations: **about 11.5 GPU-hours** and **29.2 GB** of float32 features. Throughput is
     measured for DINOv3-L; the other encoders are extrapolated.
2. **DF24 images:**
   - resolution: the 300 px release (6.42 GB) or the full-size images (about 110 GB) — sizes from the earlier
     dataset audit (v1 `docs/vlm_prototype_research/DATASET_taxonomic_transfer_metadata_audit.md`);
   - licence: the metadata says "Public Domain", the README CC BY-NC-ND 4.0. Research use only either way, but please
     confirm.
3. **FGVC-Aircraft checksum mismatch:** accept the verified-by-repetition archive, or contact the publisher.
4. **Test features:** cache them in the same pass (they stay unread until a final-evaluation authorisation, enforced
   in code), or extract train + val only now.
