#!/bin/bash
# FIN-Benthic2 fold-0 geometry replication (task.md 366ee62). Features are extracted on LUMI
# (scripts/lumi/finbenthic2_extract.sh, at the author's request); everything else runs on the Dell. Stages, in order:
#   cohort     frozen-CSV and fold-0 checks + precommitted selection (<= 200 train specimens/taxon, all val,
#              <= 10 frames/camera) + resource audit
#   pack       all FIN-Benthic2 images into one verified HDF5 (for the copy to LUMI) + cohort_hdf5.parquet
#   [LUMI]     copy the HDF5 and cohort_hdf5.parquet to /flash, run finbenthic2_extract.sh, copy the cache back to
#              $OUT/features_lumi/dinov3_l
#   verify     check the copied cache covers cohort_hdf5.parquet exactly
#   geometry   within-/between-class geometry and one-vs-several Gaussians (8 and 16 PCs); no source field
#   montage    real-image montages of the specimens the geometry stage selected (train only)
#   metadata   do the 2-component groups follow image (crop) size? (camera views: in the geometry stage)
#   compare    side-by-side summary with the AquaMonitor expanded cohort
# Outputs go to $DATA/finbenthic2_fold0 (outside git). Usage: finbenthic2_geometry.sh STAGE...
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA=${DATA:-$HOME/Documents/Research/Feature_geometry_data}
OUT=$DATA/finbenthic2_fold0
CSV=${CSV:-$HOME/Documents/Lightning_hier_transformer_embedding/01_finbenthic2_processed_5splits_taxon.csv}
IDA=${IDA:-$HOME/Documents/FIN_Benthic2/IDA}
H5=$DATA/finbenthic2_hdf5/finbenthic2_images.h5
MANIFEST=$OUT/cohort/cohort_hdf5.parquet
CACHE=$OUT/features_lumi/dinov3_l
AQUA=$DATA/expanded_cohort/geometry
PY=.venv-gpu/bin/python
export PYTHONPATH=.
for stage in "$@"; do
    case $stage in
        cohort) $PY scripts/build_finbenthic2_cohort.py --csv "$CSV" --ida "$IDA" --out "$OUT/cohort" ;;
        pack) $PY scripts/pack_finbenthic2_hdf5.py --csv "$CSV" --images "$IDA/Images" --out "$H5" --cohort "$OUT/cohort/cohort.parquet" ;;
        verify) $PY scripts/verify_feature_cache.py --manifest "$MANIFEST" --cache-dir "$CACHE" --encoder dinov3_l \
                    --chunk-size 8192 --out "$OUT/features_lumi/verify_dell.json" ;;
        geometry) $PY scripts/class_geometry_pilot.py --cohort "$OUT/cohort/cohort.parquet" --caches "$MANIFEST=$CACHE" \
                      --source-column none --out "$OUT/geometry" ;;
        montage) $PY scripts/class_geometry_montage.py --cohort "$OUT/cohort/cohort.parquet" --image-root "$IDA/Images" \
                     --selection "$OUT/geometry/selection.json" --out "$OUT/geometry" ;;
        metadata) $PY scripts/component_metadata.py --cohort "$OUT/cohort/cohort.parquet" \
                      --components "$OUT/geometry/specimen_components.csv" --modality "$OUT/geometry/per_class_modality.csv" \
                      --geometry "$OUT/geometry/per_class_geometry.csv" --fields --numeric log_image_area --out "$OUT/geometry" ;;
        compare) $PY scripts/compare_datasets_geometry.py --a "$AQUA" --a-name AquaMonitor --b "$OUT/geometry" \
                     --b-name FIN-Benthic2 --out "$OUT/compare" ;;
        *) echo "unknown stage $stage" >&2; exit 1 ;;
    esac
done
