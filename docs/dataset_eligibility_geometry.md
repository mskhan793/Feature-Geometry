# Benchmark eligibility for the geometry replication

Task: `task.md` 1ece8de, "official-split benchmark eligibility and geometry replication plan". This is a metadata
and documentation audit only. No features were extracted, no statistics were fitted, nothing was downloaded and no
remote job was run.

**Gate.** A dataset is eligible only if its original authors released train, validation and test partitions with
labels for all three, and the partitions keep independent units (specimen, barcode or observation) apart.

**Evidence.** `scripts/audit_split_eligibility.py` reads the released metadata as distributed and writes one JSON per
dataset to `docs/evidence/dataset_eligibility/`. Every count below is copied from those files. Author documentation
is quoted with its source.

## 1. Decision matrix

| | AquaMonitor Classification | FIN-Benthic2 | BIOSCAN-5M | DF24 |
|---|---|---|---|---|
| **Verdict** | Eligible (reference, done) | **Eligible** | **Eligible** | **Ineligible: no official validation** |
| Source | Original parquet + paper (earlier audit) | Ärje et al. 2020, *Signal Processing: Image Communication* 115917; data on Etsin (a11cdc26-b9d0-4af1-9285-803d65a696a3) | Gharaee et al. 2024, NeurIPS Datasets & Benchmarks (arXiv 2406.12723); CC BY 3.0 | BohemianVRA/DanishFungiDataset, split news 30.7.2024; BSD, non-commercial data |
| Official partitions | 5 folds, train/val/test | 10 "machine learning" splits, TRAIN/VAL/TEST (70/10/20 % of specimens); 10 "comparison" splits | `train` / `val` / `test` (closed world) plus unseen and held-out sets | `train` and `pubtest` only |
| Labels for every partition | Yes | Yes | Yes | Train and test yes; no validation |
| Independent unit | Specimen (`individual`) | Specimen (folder name, ≤ 50 images, 2 cameras) | Specimen record (`processid`, 1 image); split grouped by DNA barcode | Observation (`observationID`) |
| Units in more than one partition | 0 | 0 in every split | 0 barcodes shared between any two partitions | 0 observations |
| Classes | 42 | 39 taxa | 11,846 species in train | 1,604 |
| Images local | Yes (HDF5, Dell; release on LUMI) | Yes, 460,004, 0 missing (13 GB) | Yes, all 473,094 labelled images (cropped_256, 4.3 GB) | No (metadata only) |

## 2. FIN-Benthic2

**Official protocol (author text).** Ärje et al. 2020, Section 3: "We split the specimens randomly into training
(70%), validation (10%) and test (20%) data 10 times … We refer to these sets as the 'machine learning data' … For
the test sets in the machine learning data, we included all images (max. 50) per specimen." A second set of 10
"comparison data" splits has balanced test sets of 45–46 specimens (matched to a human proficiency test, at most 10
test images per specimen) and the remaining specimens split 80 % train / 20 % validation. The released files are
`Machine_learning_splits.txt` (SPLIT1–SPLIT10) and `Comparison_data_splits.txt`; the Etsin landing page did not load
for licence and version details.

**Machine-learning splits, measured** (`finbenthic2.json`):
- 460,004 images, 9,631 specimens, 39 taxa; every specimen has one taxon; 0 image files missing.
- Every split: 6,741 train / 953 val / 1,937 test specimens (the paper's Table 2 gives n = 1937 per test split);
  0 specimens in more than one partition.
- All 39 taxa are in train in every split. One taxon is missing from validation in SPLIT1, 2, 3, 8, 9
  (Hydropsyche saxonica, Sialis sp. or Capnopsis schilleri) and from test in SPLIT1 (Hydropsyche saxonica).
  **SPLIT4, 5, 6, 7 and 10 have all 39 taxa in train, val and test.**
- Train specimens per taxon (SPLIT1): min 12, median 162, max 610; 35 taxa have ≥ 20 and 16 taxa ≥ 200.
  Validation is thin: median 22.5 specimens per taxon, 31 taxa with ≥ 8.
- The 10 splits are independent random draws, so a specimen is a test specimen in some splits and not others
  (8,630 of 9,631 are test at least once). They are not a cross-validation partition and must not be pooled.
- Acquisition metadata: the image name prefix gives two cameras (`0-`: 218,384 images, `1-`: 241,620). There is no
  site, year or source field in the split files; ImageJ shape features are released separately.

**Comparison splits:** specimen-disjoint as well, but 252,392 image names in `Comparison_data_splits.txt` do not exist
in `Images/` (same number of images per specimen as the machine-learning file, different file names). Partition
membership is per specimen, so it is usable, but these splits are secondary and are not proposed here.

**The 5-fold CSV chosen in chat (`01_finbenthic2_processed_5splits_taxon.csv`, sha256 1c18f058…)** is not one of the
released splits. It was produced by the taxonomist toolkit (`StratifiedGroupKFold` by specimen, val = one inner
fold); its partition agrees with the closest released split for only 51 % of specimens in every fold, the level
expected for independent random splits. It is specimen-disjoint and uses identical images and labels, but under
this task's rule it is a derived split. **Author decision needed** (section 6).

## 3. BIOSCAN-5M

**Official protocol (author text).** Gharaee et al. 2024, Section 4.1: closed-world `train`, `val`, `test` are built
from records with a scientific species name; "for seen and unseen records we split the records by placing all samples
with the same barcode in the same partition, to ensure there is no repetition of barcodes across splits"; the test set
uses a flattened species distribution; "the val set is for model evaluation during development and hyperparameter
tuning, and the test set is for final evaluation." Use the `split` field of the metadata.

**Measured** (`bioscan5m.json`, metadata sha256 in the JSON):
- train 289,203 records / 11,846 species; val 14,757 / 3,378; test 39,373 / 3,483. These match the authors' table.
- `processid` unique; 0 barcodes shared between any two of the seven labelled partitions; every val and test species
  occurs in train.
- Train records per species: median 3; 1,588 species have ≥ 20 and 256 have ≥ 200. Species with ≥ 20 train records
  **and** at least one val record: 1,579; with ≥ 200: 256. Val records per species: median 1, only 340 species
  with ≥ 8. Test: 3–33 per species.
- Every labelled record has its image locally (cropped_256; pretrain images not downloaded and not needed).
- **Consequence:** one image per specimen, a long tail and very thin validation. Gaussian or mixture fits are
  possible only for a minority of species; most species give insufficient evidence and must be reported as such.

**Relation to earlier projects.** The BIOSCAN study in the Aqua v1 repository used these BIOSCAN-5M partitions. The
older file `~/Documents/01_bioscan_greater_100_processed_3splits_family.csv` (BIOSCAN-1M images, family label,
columns `0,1,2`) is a taxonomist-toolkit split, not BIOSCAN-1M's own split columns
(`large_diptera_family` etc.), and is not eligible.

## 4. DF24 (ineligible)

The release contains `DanishFungi2024-train.csv` and `DanishFungi2024-pubtest.csv` only. The authors' README lists a
train/test split by observation, and their training script sets `valid_metadata = args.test_path`, i.e. it validates
on the test file. Measured: train 266,273 images / 159,026 observations; pubtest 29,665 / 18,144; 1,604 classes in
both; 0 observations in both; test labels present. **No official validation partition, so DF24 fails the gate.**
Its metadata stay on the Dell; no images were downloaded.

## 5. Excluded or not benchmarks (from the 2026-11-05 audit, unchanged)

DF20 (17,059 observations in both train and test), iNat2021 Aves/Plantae (no public test labels, no observation id),
NAFlora-1M (test labels withheld, no specimen id). Høye data, the Aqua 2022 supplement, MassID45 and the fish ROIs
have no official labelled three-way protocol.

## 6. Ranking and decisions

1. **AquaMonitor Classification** — reference; the 200-specimen DINOv3-L analysis stands, no rerun.
2. **FIN-Benthic2, official machine-learning splits** — first independent replication. Imaging like
   AquaMonitor's: two cameras view a specimen in an alcohol-filled cuvette, and at most 50 images per specimen are
   kept (Ärje et al. 2020, Section 3 and Fig. 3), so the specimen-level protocol transfers unchanged.
   Proposal: one primary split with all 39 taxa in every partition, the first such being **SPLIT4**; further splits
   only as separately approved replicates, never pooled.
3. **BIOSCAN-5M closed-world** — second replication, a different imaging setup (single top view, one image per
   specimen); analysis restricted to species with enough train and val records, everything else reported as
   insufficient.

**Decisions for the author:**
- **D1 (FIN-Benthic2 partitions).** The task rule calls for the authors' released splits (machine-learning SPLITk);
  the chat decision was the taxonomist 5-fold CSV. Choose one: (a) released SPLIT4 as primary (recommended under the
  rule), or (b) the 5-fold CSV, recorded as a derived protocol and an exception to the rule.
- **D2 (BIOSCAN-5M support threshold)** for "eligible species" in the geometry fits, e.g. ≥ 20 train records and
  ≥ 1 val record (1,579 species).

## 7. Proposed first replication (not executed)

Same design as the AquaMonitor expanded cohort, frozen DINOv3-L with its official preprocessing, native 1024-D
features, PCA and all models fitted on train only, test untouched:
- **FIN-Benthic2 (SPLIT4 or the chosen protocol):** all train specimens up to 200 per taxon, all val specimens, up to
  10 spread frames per camera per specimen, one camera-balanced mean per specimen. Upper bound (6,741 + 953
  specimens × 20 frames) 153,880 images, about 45 min on the local RTX 4090 at the expanded-cohort rate (192,492
  images in 55 min 44 s), about 0.6 GB of float32 features.
- **BIOSCAN-5M:** train up to 200 records per species and all val records, one image each. Upper bound
  289,203 + 14,757 = 303,960 images, about 1.5 h at the same rate, about 1.2 GB of features. Fits only for species
  meeting D2.
- Analyses as before: within-class spectra and effective rank, between-class d′, one Gaussian vs K = 2/3 tied and
  regularised full mixtures at 8 and 16 PCs on val log density, and checks against camera (FIN-Benthic2) or
  country/collection metadata (BIOSCAN-5M) instead of biological interpretation.

Both runs fit on the Dell (49 GB free at the time of the audit). Nothing is started until the author approves.
