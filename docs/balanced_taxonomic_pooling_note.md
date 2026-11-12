# Class-balanced and taxonomy-informed covariance pooling — provenance, taxonomy feasibility and design (task.md f54f534)

## Published sources (reused unchanged)

| Source | Local copy | SHA256 |
|---|---|---|
| Ledoit & Wolf, JMVA 2004 (preprint Feb 2001) | `~/Documents/Research/methods/ledoit_wolf2004/paper.pdf` | `08e79244bf0ea47acb4a35189cfd7614bbc21cc058684da3d0cfe50d83717f1c` |
| Raninen & Ollila, IEEE TSP 69:5681–5692, 2021 (doi 10.1109/TSP.2021.3118546) | `~/Documents/Research/methods/Shrinkage/Aalto_thesis/Coupled_Regularized_…Multiple_Classes.pdf` | `14853bade1f89a1d3a4e0645569871bc3cb2ef7de3e5b50c6c6f9c6aab0f97cf` |
| Authors' code CoupledRSCM @ `bb09e12`, `Python/multiclass_rscm.py` (MIT), vendored in `third_party/coupled_rscm/` | — | `a67e443d18613b2671f6997059cf900a3a51853d5584f7251e6a290b780aa326` |

Method details, the one essential adaptation (`reshape(4)` → `reshape(K)`) and the paper-vs-code check:
`docs/coupled_rscm_source_note.md`. No new paper is used here.

**Our proposals (not from these papers):** class-balanced pooling, leave-target-out pools, genus → family → global
donor selection, the degrees-of-freedom support rule and the nearest-mean control. All are defined in
`src/methods/taxonomic_pooling.py` and `scripts/balanced_taxonomic_pooling.py`, fixed in commit `ff2cdf3` before the
first run (LUMI job `23686150`).

## Taxonomy sources (dataset authors' own tables)

- **AquaMonitor** — release `mikkoim/aquamonitor` rev `9a30834`. The class is `taxon_group`. `taxon_mapping.csv`
  (git blob `035f202e6650a597db4a8fd63059dbf5d67b4947`) gives its scientific `taxon`, and `taxon_table.csv` (blob
  `f0c57504a990b489f4511349e9d051887849865e`) gives kingdom…species. `README-metadata.md` documents both. Local copy:
  `~/Documents/Research/Feature_geometry_data/aquamonitor_release_metadata/`; LUMI copy in the verified release
  download. The blob hashes are equal locally, on LUMI and in the release manifest.
- **FIN-Benthic2** — the release's `IDA/Table of classes.pdf` (Table I; SHA256
  `cd58c251f2ad5954caca692bb3800fb058923e41e91fabec4c47a557ec380de2`). It is transcribed verbatim by
  `scripts/make_finbenthic2_taxonomy.py` to `finbenthic2_table_of_classes.csv` (SHA256
  `e0b1580810ac0613ae79a10dd283ffcfc7d0116e39991120cbd708958d380896`; copy in
  `docs/evidence/balanced_taxonomic_pooling/taxonomy_audit/`; LUMI copy on `/flash`, same hash). Class labels are
  matched to its 'Taxa' column after one fixed normalisation (underscores → spaces, trailing '.' removed); the match
  must be one-to-one.

## Feasibility audit (gate A2; `docs/evidence/balanced_taxonomic_pooling/taxonomy_audit/`)

- **FIN-Benthic2:** all 39 classes map one-to-one. For every class, the specimen count in our selection equals the
  release table's #specimens. Ranks: 26 species, 12 genus (the "sp." taxa), 1 family (Simuliidae).
- **Release quirks (kept verbatim):**
  - misspelled names: *Perlodiae*, *Taenioptegyridae*, *Sphaeridae* (families) and *Rhycophila* (genus);
  - repeated class numbers (taxa 36, family 8, genus 28, species 25), so grouping uses names, never numbers.
- **AquaMonitor:** 40 of 42 classes have ancestry. *Oligochaeta* and *Turbellaria* are not in `taxon_table`
  → global fallback.
  - Ranks: 23 species, 13 genus, 4 family.
  - Two classes share one species (*Oulimnius tuberculatus*, larva and adult). In the PCA spaces each is the other's
    genus-level donor. In native space only the adult does (the larva falls back to global under the df rule).
  - The table puts the subfamily *Tanypodinae* in its genus column and *Hydrachnidia* in its family column; used as
    given.
- **Independent cross-check:** the vision-language project's maps (v1
  `assets/label_maps/htc_labels/aqua_plus_finb1_finb2_hoye_vocab/taxon_parent_mappings.json`, SHA256 `4ced70a5…`;
  FIN-Benthic2 per-image table `finb2_taxon.parquet`, SHA256 `1ff288a3…`).
  - AquaMonitor: 99 of 99 links agree.
  - FIN-Benthic2 genus and family groupings: identical. Only spellings differ (the release's typos above, and v1's
    *Plectronemia*).
  - Output: `taxonomy_audit/v1_crosscheck.{csv,json}`.
- **Classes with no genus or family mate among the candidates:** 12 FIN-Benthic2 and 17 AquaMonitor classes. These
  always use the global fallback.

## Predeclared design

- **Shared protocol** (identical to the earlier coupled-RSCM runs; `src/statistics/heldout_density.py`):
  - unchanged specimen embeddings and train-fitted PCA (16, 64);
  - native coordinates without one coordinate (`native_alt` drops the other end);
  - train class means;
  - eligible targets: ≥ 15 train and ≥ 5 val specimens;
  - test rows dropped before the feature lookup.
- **One rule for every pool P:** T = β_c S_c + (1 − β_c) P and Σ = α T + (1 − α)(tr T/d) I, where:
  - β_c = the authors' CRSCM β of class c (a fixed diagnostic; tuned only for their own pool);
  - α = 1 − α_LW,c;
  - S_j = unbiased within-class covariances (authors' `scov`).
- **Pools:**
  - `pool_spec`: Σ (n_j/N) S_j over the eligible classes (= the authors' pooled SCM);
  - `pool_bal`: the same classes, equal weights;
  - `pool_bal_loo`: equal weights over all other donor candidates (≥ 15 train specimens), target left out;
  - `pool_tax`: taxonomic donors, each class once, target left out;
  - `pool_near`: the nearest-mean control.
- **Donor levels:** species-rank targets try genus then family; genus-rank targets try family; family-rank or
  unknown targets use global. A level is accepted if Σ_j (n_j − 1) ≥ d.
- **Nearest-mean control:** as many nearest classes (by train-mean distance) as there are taxonomic donors, extended
  until the same df rule holds. Defined only where a taxonomic level was accepted.
- **Contrasts:**
  - balancing = `pool_bal` − `pool_spec`;
  - taxonomy = `pool_tax` − `pool_bal_loo`;
  - taxonomy_vs_nearest = `pool_tax` − `pool_near`;
  - leave_target_out = `pool_bal_loo` − `pool_bal`;
  - each pool vs LW;
  - coupled vs LW (reproduction).
- **Reproduction checks:** LW, coupled and `pool_spec` (= the earlier `pooling_lw_shrink` cell) must equal the
  earlier runs exactly.
