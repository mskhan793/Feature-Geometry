#!/bin/bash
# Camera-conditioned Gaussian pilot, local Dell run (task.md 2026-11-07). Stages, in order:
#   inventory  camera inventory of the benchmark manifest (read-only)
#   cohort     capped fold0 train/val cohort
#   extract    DINOv3-L features of the cohort images (official v1 preprocessing) into features_$ACCEL, with wall time
#   compare    features_gpu vs features_cpu on the chunks both hold
#   analyse    Gaussian model comparison and supporting evidence (on features_$ACCEL)
# ACCEL=gpu (default; .venv-gpu, torch 2.10.0+cu128) or cpu (.venv, torch 2.10.0+cpu).
# Outputs go to $DATA/camera_pilot (outside git). Usage: [ACCEL=cpu] camera_pilot.sh STAGE...
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA=${DATA:-$HOME/Documents/Research/Feature_geometry_data}
OUT=$DATA/camera_pilot
MANIFEST=$DATA/manifests/aquamonitor_classif_manifest.parquet
HDF5=${HDF5:-$HOME/Documents/aqua_data/aquamonitor_complete.h5}
ACCEL=${ACCEL:-gpu}
case $ACCEL in gpu) PY=.venv-gpu/bin/python ;; cpu) PY=.venv/bin/python ;; *) echo "ACCEL must be gpu or cpu" >&2; exit 1 ;; esac
FEATURES=$OUT/features_$ACCEL
for stage in "$@"; do
    case $stage in
        inventory) $PY scripts/camera_inventory.py --manifest "$MANIFEST" --out "$OUT/inventory" ;;
        cohort) $PY scripts/build_camera_cohort.py --manifest "$MANIFEST" --inventory "$OUT/inventory/inventory.json" --out "$OUT/cohort" ;;
        extract) /usr/bin/time -v $PY scripts/extract_features.py --manifest "$OUT/cohort/cohort.parquet" --hdf5 "$HDF5" \
                     --encoder dinov3_l --cache-dir "$FEATURES" --chunk-size 2048 --batch-size 32 --num-workers 8 \
                     --accelerator "$ACCEL" ;;
        compare) $PY scripts/compare_feature_caches.py --a "$OUT/features_gpu/dinov3_l" --b "$OUT/features_cpu/dinov3_l" \
                     --out "$OUT/features_gpu_vs_cpu.json" ;;
        analyse) $PY scripts/camera_gaussian_pilot.py --cohort "$OUT/cohort/cohort.parquet" --cache "$FEATURES/dinov3_l" \
                     --out "$OUT/analysis_$ACCEL" ;;
        *) echo "unknown stage $stage" >&2; exit 1 ;;
    esac
done
