# Coding assistant response — Feature-Geometry

**Current task:** `task.md` 1ece8de, "official-split benchmark eligibility and geometry replication plan".
**Status: done, stopped for the author's review.**
- **Ran:** a read-only audit of released metadata and author documentation on the Dell (`scripts/audit_split_eligibility.py`).
- **Not done:** no feature extraction, no statistics, no downloads, nothing remote, no test scoring.

**Previous result:** expanded-cohort geometry, response at commit 83f6faa, evidence in `docs/evidence/expanded_geometry/`
(conclusion unchanged: tied K=2/K=3 mixtures beat one Gaussian at n ≤ 200, source-specific Gaussians do better still;
acquisition and biology are not disentangled).

## 1. Answer in brief

Full report: [`docs/dataset_eligibility_geometry.md`](docs/dataset_eligibility_geometry.md). Evidence:
`docs/evidence/dataset_eligibility/{finbenthic2,bioscan5m,df24}.json`.

| Dataset | Verdict | Why |
|---|---|---|
| AquaMonitor Classification | Eligible, reference | Unchanged; no rerun |
| **FIN-Benthic2** | **Eligible** | Authors' 10 "machine learning" splits, TRAIN/VAL/TEST by specimen (Ärje et al. 2020); 6,741 / 953 / 1,937 specimens per split, 0 specimens in two partitions, all 460,004 images on the Dell |
| **BIOSCAN-5M** | **Eligible** | Authors' closed-world `train`/`val`/`test` (Gharaee et al. 2024), 0 DNA barcodes shared across partitions, labels public, all 473,094 labelled images on the Dell |
| DF24 | Ineligible | Release has train and test only; the authors' own training script validates on the test file |
| DF20, iNat2021, NAFlora-1M | Excluded | As in the 2026-11-05 audit |

**Correction to what I said in chat on 2026-11-10:** FIN-Benthic2's `Machine_learning_splits.txt` does contain VAL in
all 10 splits; I had only looked at its first rows.

## 2. Points to know

- FIN-Benthic2's 10 splits are independent random draws, not cross-validation folds; they must not be pooled. SPLIT4,
  5, 6, 7 and 10 have all 39 taxa in train, val and test; the others miss one rare taxon in val (or test, SPLIT1).
- FIN-Benthic2 validation is small (median 22.5 specimens per taxon); BIOSCAN-5M validation is very thin (median 1
  record per species; 1,579 species have ≥ 20 train records and a val record).
- `Comparison_data_splits.txt` (FIN-Benthic2's second protocol) names 252,392 images that are not in `Images/`; its
  specimen assignments are intact. Not proposed for use.
- The FIN-Benthic2 5-fold CSV from the taxonomist toolkit agrees with the closest released split for 51 % of
  specimens, i.e. it is an independent derived split.

## 3. Decisions needed

- **D1, FIN-Benthic2 partitions:** (a) released machine-learning SPLIT4 as primary (recommended under the task's
  rule) or (b) the 5-fold taxonomist CSV chosen in chat, recorded as a derived protocol and an exception.
- **D2, BIOSCAN-5M support threshold** for species entering the geometry fits (e.g. ≥ 20 train and ≥ 1 val record).
- **Approval to run** the first replication proposed in section 7 of the report (local RTX 4090; FIN-Benthic2 upper
  bound 153,880 images, about 45 min; BIOSCAN-5M upper bound 303,960 images, about 1.5 h).

## 4. Unknowns

- FIN-Benthic2 licence and version: the Etsin landing page did not render for the audit.
- BIOSCAN-5M is grouped by barcode, not by BIN; specimens with different barcodes of one BIN can sit in different
  partitions (allowed by the authors' design).
