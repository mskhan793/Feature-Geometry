#!/bin/bash
# Five-fold train->validation geometry (task.md 92a605a, output B) on LUMI CPU nodes: one array task per fold
# (SLURM_ARRAY_TASK_ID = fold index). Per fold: class_geometry_pilot.py (fold's train specimens fitted, val scored,
# test rows never loaded) and component_metadata.py, on the full-specimen selection and its verified LUMI caches.
#
# Usage: sbatch --array=0-4 [--cpus-per-task N --mem M --time T] geometry_folds.sh CODE_DIR DATASET OUT_ROOT
#   DATASET: finbenthic2 | aquamonitor; results in OUT_ROOT/DATASET/geometry/foldK.
#SBATCH --job-name=fg_geom
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -eo pipefail  # no -u: the module scripts read unset variables

export CODE=$1 DATASET=$2 ROOT=$3
FOLD=fold$SLURM_ARRAY_TASK_ID
echo "start $(date -Is) job ${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID node $SLURM_NODELIST $DATASET $FOLD"
source "$CODE/scripts/lumi/geometry_inputs.sh"
echo "inputs ok $(date -Is) code $(cat "$CODE/CODE_COMMIT")"
OUT=$ROOT/$DATASET/geometry/$FOLD
mkdir -p "$OUT"
ENV="export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; cd $CODE"
singularity exec "$SIF" bash -c "$ENV; python scripts/class_geometry_pilot.py --cohort $COHORT --caches $CACHES \
    --out $OUT --fold $FOLD --source-column $SOURCE --jobs $SLURM_CPUS_PER_TASK > $OUT/geometry.log 2>&1"
singularity exec "$SIF" bash -c "$ENV; python scripts/component_metadata.py --cohort $COHORT \
    --components $OUT/specimen_components.csv --modality $OUT/per_class_modality.csv \
    --geometry $OUT/per_class_geometry.csv --out $OUT --fold $FOLD $META_ARGS > $OUT/metadata.log 2>&1"
echo "end $(date -Is)"
