#!/bin/bash
# FIN-Benthic2 fold-0 geometry replication (task.md 366ee62), local Dell run. Stages, in order:
#   cohort     frozen-CSV and fold-0 checks + precommitted selection (<= 200 train specimens/taxon, all val,
#              <= 10 frames/camera) + resource audit
#   extract    DINOv3-L features of every cohort image (GPU, .venv-gpu, TF32 off)
#   verify     check the cache covers cohort.parquet exactly
#   cpucheck   first 256 cohort images again on the CPU (.venv) and compared with the GPU cache
#   geometry   within-/between-class geometry and one-vs-several Gaussians (8 and 16 PCs); no source field
#   montage    real-image montages of the specimens the geometry stage selected (train only)
#   metadata   do the 2-component groups follow camera views, imaging runs or image (crop) size?
#   compare    side-by-side summary with the AquaMonitor expanded cohort
# Outputs go to $DATA/finbenthic2_fold0 (outside git). Usage: finbenthic2_geometry.sh STAGE...
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA=${DATA:-$HOME/Documents/Research/Feature_geometry_data}
OUT=$DATA/finbenthic2_fold0
CSV=${CSV:-$HOME/Documents/Lightning_hier_transformer_embedding/01_finbenthic2_processed_5splits_taxon.csv}
IDA=${IDA:-$HOME/Documents/FIN_Benthic2/IDA}
AQUA=$DATA/expanded_cohort/geometry
PY=.venv-gpu/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=.
for stage in "$@"; do
    case $stage in
        cohort) $PY scripts/build_finbenthic2_cohort.py --csv "$CSV" --ida "$IDA" --out "$OUT/cohort" ;;
        extract) /usr/bin/time -v $PY scripts/extract_features.py --manifest "$OUT/cohort/cohort.parquet" \
                     --image-root "$IDA/Images" --encoder dinov3_l --cache-dir "$OUT/features_gpu" --chunk-size 8192 \
                     --batch-size 64 --num-workers 8 --accelerator gpu ;;
        verify) $PY scripts/verify_feature_cache.py --manifest "$OUT/cohort/cohort.parquet" \
                    --cache-dir "$OUT/features_gpu/dinov3_l" --encoder dinov3_l --chunk-size 8192 --out "$OUT/features_gpu/verify.json" ;;
        cpucheck) .venv/bin/python scripts/extract_features.py --manifest "$OUT/cohort/cohort.parquet" --image-root "$IDA/Images" \
                      --encoder dinov3_l --cache-dir "$OUT/features_cpu256" --chunk-size 8192 --limit-rows 256 \
                      --batch-size 32 --num-workers 4 --accelerator cpu
                  $PY scripts/compare_cache_prefix.py --manifest "$OUT/cohort/cohort.parquet" --full "$OUT/features_gpu/dinov3_l" \
                      --prefix "$OUT/features_cpu256/dinov3_l" --rows 256 --out "$OUT/features_gpu_vs_cpu256.json" ;;
        geometry) $PY scripts/class_geometry_pilot.py --cohort "$OUT/cohort/cohort.parquet" \
                      --caches "$OUT/cohort/cohort.parquet=$OUT/features_gpu/dinov3_l" --source-column none --out "$OUT/geometry" ;;
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
