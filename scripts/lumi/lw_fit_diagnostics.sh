#!/bin/bash
# Ledoit-Wolf diagnostics (frames per specimen, specimens per class, per-class shrinkage), one array task per dataset
# and fold on LUMI CPU nodes: SLURM_ARRAY_TASK_ID 0-4 = FIN-Benthic2 fold0-4, 5-9 = AquaMonitor fold0-4. Existing
# full-specimen DINOv3-L caches (geometry_inputs.sh).
#
# Usage: sbatch --array=0-9 lw_fit_diagnostics.sh CODE_DIR OUT_ROOT
#SBATCH --job-name=fg_lwfit
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

export CODE=$1 ROOT=$2
DATASETS=(finbenthic2 aquamonitor)
export DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID / 5))]}
FOLD=fold$((SLURM_ARRAY_TASK_ID % 5))
echo "start $(date -Is) job ${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID node $SLURM_NODELIST $DATASET $FOLD"
source "$CODE/scripts/lumi/geometry_inputs.sh"
echo "inputs ok $(date -Is) code $(cat "$CODE/CODE_COMMIT")"
OUT=$ROOT/$DATASET/$FOLD
mkdir -p "$OUT"
singularity exec "$SIF" bash -c "export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; cd $CODE; \
    python scripts/lw_fit_diagnostics.py --cohort $COHORT --caches $CACHES --fold $FOLD --out $OUT > $OUT/run.log 2>&1"
echo "end $(date -Is)"
