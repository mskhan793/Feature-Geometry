#!/bin/bash
# Bhattacharyya class overlap vs d' on fold 0 of FIN-Benthic2 and AquaMonitor (task.md 0237ac7), on one LUMI CPU node:
# existing full-specimen DINOv3-L caches (scripts/lumi/geometry_inputs.sh), fold-0 train and val specimens only; the
# corrected fold-0 geometry run (GEOM_ROOT/<dataset>/geometry/fold0) supplies the reference d' and the per-class labels.
#
# Usage: sbatch bhattacharyya_fold0.sh CODE_DIR GEOM_ROOT OUT_ROOT
#SBATCH --job-name=fg_bhatt
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail  # no -u: the module scripts read unset variables

export CODE=$1 GEOM=$2 OUT=$3
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST"  # before any setup
for DATASET in finbenthic2 aquamonitor; do
    export DATASET
    source "$CODE/scripts/lumi/geometry_inputs.sh"
    echo "inputs ok $DATASET $(date -Is) code $(cat "$CODE/CODE_COMMIT")"
    G=$GEOM/$DATASET/geometry/fold0
    mkdir -p "$OUT/$DATASET"
    singularity exec "$SIF" bash -c "export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; \
        cd $CODE; python scripts/bhattacharyya_overlap.py --cohort $COHORT --caches $CACHES --out $OUT/$DATASET \
        --fold fold0 --pcs 64 8 16 --existing-pairs $G/class_pairs.csv --labels $G/per_class_modality.csv \
        > $OUT/$DATASET/run.log 2>&1"
    echo "$DATASET done $(date -Is)"
done
echo "end $(date -Is)"
