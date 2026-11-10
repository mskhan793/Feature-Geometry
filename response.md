# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 2f6774c / e5f42d9, "freeze FIN-Benthic2 five-split and DF24 derived-validation protocols".
**Status: done, stopped for the author's review.**
- **Ran:** metadata-only checks on the Dell; one deterministic DF24 validation draw from official train.
- **Not done:** no image downloads, no weights, no GPU extraction, no Gaussian fits, no classifiers, nothing remote,
  no test evaluation.

**Full details:** [`docs/dataset_eligibility_geometry.md`](docs/dataset_eligibility_geometry.md),
**section 0** (the frozen FIN-Benthic2 and DF24 protocols, all counts and checks) and **section 7** (the proposed
replication plan with image counts and run-time estimates). This file is the short summary.

**Previous results:** eligibility audit, response at commit b04f4ae (`docs/dataset_eligibility_geometry.md`,
sections 1–6 kept for provenance); expanded-cohort geometry at 83f6faa.

## 1. Answer in brief

| Dataset | Status | Protocol |
|---|---|---|
| AquaMonitor Classification | Eligible, official | Original five folds (reference) |
| **FIN-Benthic2** | **Eligible, author-approved derived protocol** | Five-split CSV, sha256 `1c18f058…`; not a publisher split |
| BIOSCAN-5M | Eligible, official | Closed-world train / val / test |
| **DF24** | **Eligible, author-approved derived validation** | Official train minus derived val (seed 20261110); official pubtest unchanged |

Evidence: `docs/evidence/split_protocols/`.

## 2. FIN-Benthic2 five-split (frozen, not regenerated)

- Checked with `scripts/check_finbenthic2_folds.py` (`src/data/finbenthic2_folds.py`): 460,004 images, 9,631
  specimens, 39 taxa, identical images and labels to the released files, 0 image files missing.
- Each fold: about 6,160 train / 1,540 val / 1,926 test specimens; 0 specimens in two partitions; 0 test specimens
  seen in that fold's train or val; all 39 taxa in every partition (smallest class 8–12 train, 1–2 val, 1–4 test
  specimens). Per-class counts: `finbenthic2_fivefold_class_support.csv`.
- **How the folds were formed:** taxonomist `StratifiedGroupKFold` by specimen on `taxon`; test = outer fold, val =
  one inner fold of the remaining specimens. **The test sets form a disjoint five-fold partition** (every specimen is
  test exactly once, 0 overlap between folds). Validation sets overlap between folds (205–266 specimens per pair),
  which is allowed.

## 3. DF24 derived validation (one draw, committed)

- `scripts/derive_df24_validation.py` with `src/data/df24_validation.py`; seed 20261110; unit `observationID`;
  near-duplicate groups (same recorder, species, date, locality on any image, linked by union-find; 159 observations
  have two recorders) kept on one side; about 10 % per species; species with fewer than 10 observations train-only.
- Train 143,155 observations / 239,727 images; derived val 15,871 / 26,546 (1,591 species); pubtest 18,144 /
  29,665, unchanged. 1,591 species hit their target exactly, 13 rare species (4–9 observations) have no val. The
  near-duplicate constraint was satisfiable for every species.
- All integrity checks pass; the saved assignment (`df24_split_assignment.csv.gz`, sha256 `688564d2…`) is
  byte-identical on a rerun. Source hashes and the rule are in `df24_derived_validation.json`.
- **Limitation inherited from the official split:** 713 near-duplicate keys appear on both the train side and
  pubtest, 68 on both derived val and pubtest (the official train/test boundary; reported, not changed).

## 4. Next (needs approval)

Proposed order (`docs/dataset_eligibility_geometry.md`, section 7): FIN-Benthic2 fold0 (≤ 154,140 images, about 45 min local GPU), then
BIOSCAN-5M (≤ 303,960 images, about 1.5 h; **D2, the species support threshold, is still open**), then DF24 after
an image download (about 6.4 GB) and storage approval (≤ 266,273 images, about 1.3 h).

## 5. Code and tests

- New: `src/data/finbenthic2_folds.py`, `src/data/df24_validation.py`, `scripts/check_finbenthic2_folds.py`,
  `scripts/derive_df24_validation.py`, `tests/test_split_protocols.py` (synthetic metadata only: grouping, seeded
  determinism, rare classes, leak and bad-value detection). Full suite: 49 passed.
