#!/bin/bash
# Expanded-cohort geometry (task.md fb46e1f), local Dell run. Stages, in order:
#   cohort     audit + precommitted selection (up to 200 train specimens/class, all val, 10 frames/camera) + estimate
#   extract    DINOv3-L features of the images not already in the camera-pilot cache (GPU, .venv-gpu, TF32 off)
#   verify     check the new cache covers new_images.parquet exactly
#   geometry   within-/between-class geometry and one-vs-several Gaussians (8 and 16 PCs) on the expanded cohort
#   montage    real-image montages of the specimens the geometry stage selected (train geometry only)
#   metadata   do the 2-component groups follow dataset / year / site?
#   compare    expanded vs the 30-specimen run (verdicts, nearest pairs, centroid and subspace stability)
# Outputs go to $DATA/expanded_cohort (outside git). Usage: expanded_geometry.sh STAGE...
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA=${DATA:-$HOME/Documents/Research/Feature_geometry_data}
OUT=$DATA/expanded_cohort
PILOT=$DATA/camera_pilot
MANIFEST=$DATA/manifests/aquamonitor_classif_manifest.parquet
HDF5=${HDF5:-$HOME/Documents/aqua_data/aquamonitor_complete.h5}
PY=.venv-gpu/bin/python
CACHES=("$PILOT/cohort/cohort.parquet=$PILOT/features_gpu/dinov3_l" "$OUT/cohort/new_images.parquet=$OUT/features_gpu/dinov3_l")
for stage in "$@"; do
    case $stage in
        cohort) $PY scripts/build_expanded_cohort.py --manifest "$MANIFEST" --reuse-cohort "$PILOT/cohort/cohort.parquet" \
                    --reuse-cache "$PILOT/features_gpu/dinov3_l" --out "$OUT/cohort" ;;
        extract) /usr/bin/time -v $PY scripts/extract_features.py --manifest "$OUT/cohort/new_images.parquet" --hdf5 "$HDF5" \
                     --encoder dinov3_l --cache-dir "$OUT/features_gpu" --chunk-size 8192 --batch-size 32 --num-workers 8 \
                     --accelerator gpu ;;
        verify) $PY scripts/verify_feature_cache.py --manifest "$OUT/cohort/new_images.parquet" --cache-dir "$OUT/features_gpu/dinov3_l" \
                    --encoder dinov3_l --chunk-size 8192 --out "$OUT/features_gpu/verify.json" ;;
        geometry) $PY scripts/class_geometry_pilot.py --cohort "$OUT/cohort/cohort.parquet" --caches "${CACHES[@]}" \
                      --out "$OUT/geometry" ;;
        montage) $PY scripts/class_geometry_montage.py --cohort "$OUT/cohort/cohort.parquet" --hdf5 "$HDF5" \
                     --selection "$OUT/geometry/selection.json" --out "$OUT/geometry" ;;
        metadata) $PY scripts/component_metadata.py --cohort "$OUT/cohort/cohort.parquet" \
                      --components "$OUT/geometry/specimen_components.csv" --modality "$OUT/geometry/per_class_modality.csv" \
                      --geometry "$OUT/geometry/per_class_geometry.csv" --out "$OUT/geometry" ;;
        compare) $PY scripts/compare_geometry_runs.py --old "$PILOT/geometry_gpu" --old-cohort "$PILOT/cohort/cohort.parquet" \
                     --old-cache "$PILOT/features_gpu/dinov3_l" --new "$OUT/geometry" --new-cohort "$OUT/cohort/cohort.parquet" \
                     --new-caches "${CACHES[@]}" --out "$OUT/compare" ;;
        *) echo "unknown stage $stage" >&2; exit 1 ;;
    esac
done
