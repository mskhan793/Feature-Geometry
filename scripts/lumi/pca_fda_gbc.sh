#!/bin/bash
# PCA-GBC vs regularised FDA-GBC (task.md eeb9707), one array task per dataset and fold on LUMI CPU nodes:
# SLURM_ARRAY_TASK_ID 0-4 = FIN-Benthic2 fold0-4, 5-9 = AquaMonitor fold0-4. Existing full-specimen DINOv3-L caches
# (geometry_inputs.sh); fold 0 is checked against the earlier Bhattacharyya run (PREV/<dataset>).
#
# Usage: sbatch --array=0-9 pca_fda_gbc.sh CODE_DIR PREV OUT_ROOT
#SBATCH --job-name=fg_pcafda
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -eo pipefail  # no -u: the module scripts read unset variables

export CODE=$1 PREV=$2 ROOT=$3
DATASETS=(finbenthic2 aquamonitor)
export DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID / 5))]}
FOLD=fold$((SLURM_ARRAY_TASK_ID % 5))
echo "start $(date -Is) job ${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID node $SLURM_NODELIST $DATASET $FOLD"
source "$CODE/scripts/lumi/geometry_inputs.sh"
echo "inputs ok $(date -Is) code $(cat "$CODE/CODE_COMMIT")"
OUT=$ROOT/$DATASET/$FOLD
mkdir -p "$OUT"
EXTRA=""
if [ "$FOLD" = fold0 ]; then EXTRA="--previous $PREV/$DATASET"; fi  # not $(test && echo): fails under set -e
singularity exec "$SIF" bash -c "export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; cd $CODE; \
    python scripts/pca_fda_gbc.py --cohort $COHORT --caches $CACHES --fold $FOLD --out $OUT --dims 8 16 32 $EXTRA \
    > $OUT/run.log 2>&1"
echo "end $(date -Is)"
