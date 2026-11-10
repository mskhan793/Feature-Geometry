# Benchmark eligibility for the geometry replication

Task: `task.md` 1ece8de, "official-split benchmark eligibility and geometry replication plan". This is a metadata
and documentation audit only. No features were extracted, no statistics were fitted, nothing was downloaded and no
remote job was run.

## 0. Status after the author's decisions (task 2f6774c / e5f42d9, 2026-11-10)

| Dataset | Status | Protocol in use | Frozen record |
|---|---|---|---|
| AquaMonitor Classification | Eligible, official | Original five folds; expanded-cohort result is the reference | earlier evidence |
| **FIN-Benthic2** | **Eligible by author-approved derived protocol** | The author's five-split CSV (`01_finbenthic2_processed_5splits_taxon.csv`, sha256 `1c18f058…`), made with the taxonomist toolkit (StratifiedGroupKFold by specimen). **Not a dataset-publisher split.** The released 10 machine-learning splits remain a possible later sensitivity check | `docs/evidence/split_protocols/finbenthic2_fivefold.json`, `…_class_support.csv` |
| **BIOSCAN-5M** | Eligible, official | Closed-world `train` / `val` / `test` | `docs/evidence/dataset_eligibility/bioscan5m.json` |
| **DF24** | **Eligible by author-approved derived validation** | Official `DanishFungi2024-train.csv` minus a derived validation set (seed 20261110), official `pubtest` unchanged. **Validation is not official and not comparable with published validation numbers** | `docs/evidence/split_protocols/df24_*` |

**FIN-Benthic2 five-split, measured** (`scripts/check_finbenthic2_folds.py`):
- 460,004 images, 9,631 specimens, 39 taxa; images, specimens and taxa identical to the released files; 0 image files
  missing.
- Every fold: about 6,160 train / 1,540 val / 1,926 test specimens (fold0: 6,165 / 1,542 / 1,924); 0 specimens in
  more than one partition; 0 test specimens seen in train or val; all 39 taxa in every partition. Smallest class:
  8–12 train specimens, 1–2 val, 1–4 test.
- **The test sets are a disjoint five-fold partition:** each specimen is a test specimen in exactly one fold
  (0 overlap between any two folds). Validation sets are not disjoint across folds (205–266 shared specimens per
  pair), which is allowed: val is drawn from each fold's non-test specimens.

**DF24 derived validation, measured** (`scripts/derive_df24_validation.py`, `src/data/df24_validation.py`):
- Rule: unit `observationID`; observations sharing recorder, species, date and locality on any image are linked
  (union-find) and kept on one side; per species about 10 % of observations, drawn group by group in seeded order
  without exceeding the target; species with fewer than 10 training observations stay train-only.
- Result: train 143,155 observations / 239,727 images / 1,604 species; derived val 15,871 / 26,546 / 1,591;
  pubtest unchanged, 18,144 / 29,665 / 1,604. 1,591 species reached their target exactly; 13 species (4–9
  observations) are train-only. Val observations per species: median 5, range 1–101.
- 6,869 training observations are in multi-observation near-duplicate groups (155,340 groups in all); no group is
  split. All checks pass (each observation assigned once, images preserved, no observation in two partitions, labels
  present, pubtest byte-identical). A rerun gives a byte-identical assignment (sha256 `688564d2…`).
- **Inherited limitation of the official split:** 713 near-duplicate keys (recorder, species, date, place) occur in
  both the train side and pubtest, and 68 in both the derived val and pubtest. These cross the official train/test
  boundary set by the dataset authors and are reported, not changed.

The sections below are the original audit (task 1ece8de) and are kept unchanged for provenance, except where marked.

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

*Marked 2026-11-10:* D1 resolved by the author in favour of option (b), the five-split CSV (section 0). DF24 joined
by derived validation (section 0). D2 is still open.

## 7. Proposed geometry replication plan (revised 2026-11-10; not executed)

Common design, as in the AquaMonitor expanded cohort: frozen DINOv3-L with its official preprocessing; every image
embedding cached with its image name, unit id, label and partition (native 1024-D, float32, resumable verified
chunks outside Git); one embedding per independent unit for fitting and scoring; train-fitted centring and PCA;
single regularised full Gaussian vs K = 2/3 tied and regularised full mixtures on val log predictive density at 8
and 16 PCs; native-space spectra, effective rank and between-class d′ reported separately from the PCA density
results; per-class support reported and ineligible fits marked, no class dropped silently; test never read.
Image-count estimates use the expanded-cohort rate (192,492 images in 55 min 44 s on the local RTX 4090); feature
storage is 4 KB per image.

1. **FIN-Benthic2 first, fold0 of the five-split CSV.** Unit: specimen (`individual`). Train: up to 200 specimens per
   taxon within fold0 train; val: all 1,542 fold0 val specimens. Up to 10 spread frames per camera per specimen
   (cameras from the `0-`/`1-` image prefix), one camera-balanced mean per specimen. Upper bound
   (6,165 + 1,542) × 20 = 154,140 images, about 45 min, about 0.6 GB. Folds 1–4 only as later approved replicates;
   their test sets are disjoint, so a later locked test evaluation could pool them as in AquaMonitor.
2. **BIOSCAN-5M second, official closed world.** Unit: specimen record (one image). Train: up to 200 records per
   species; val: all 14,757 records. Fits only for species meeting D2 (open). Upper bound
   289,203 + 14,757 = 303,960 images, about 1.5 h, about 1.2 GB. Acquisition checks against country and collection
   fields instead of camera.
3. **DF24 third, after image download and storage approval.** Unit: observation; one mean embedding over an
   observation's images. Train: up to 200 observations per species from the derived train side; val: all 15,871
   derived-val observations; pubtest untouched. Upper bound 239,727 + 26,546 = 266,273 images, about 1.3 h,
   about 1.1 GB. Images are not on the Dell (the 2026-11-05 audit lists about 6.4 GB at 300 px for the DF20 images,
   which DF24 reuses). Acquisition checks against recorder, substrate, habitat and year.

All three fit on the Dell (49 GB free at the time of the audit). Nothing is started until the author approves.
