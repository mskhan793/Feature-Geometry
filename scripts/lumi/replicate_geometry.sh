#!/bin/bash
# 8 x 4 geometry replication (task.md a82e89e) on LUMI CPU nodes: array task i runs row i of a task table made by
# scripts/replication/make_tasks.py (encoder, dataset, partition, analysis, cohort, caches, extra arguments). Results go
# to OUT_ROOT/<encoder>/<dataset>/<partition>/<analysis>; a finished task writes DONE there and is skipped if rerun.
# Every cohort/cache path is checked inside the container first. No GPU, no test rows (fold test rows are dropped
# before any feature lookup; single-protocol cohorts contain none).
#
# Usage: sbatch --array=1-N [--cpus-per-task C --mem M --time T] replicate_geometry.sh CODE_DIR TASKS.tsv OUT_ROOT
#SBATCH --job-name=fg_repl
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -eo pipefail  # no -u: the module scripts read unset variables

export CODE=$1 TASKS=$2 ROOT=$3
echo "start $(date -Is) job ${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID node $SLURM_NODELIST code $(cat "$CODE/CODE_COMMIT")"
source "$CODE/scripts/lumi/container_env.sh"
IFS=$'\t' read -r ENC DATASET PART ANALYSIS COHORT CACHES EXTRA META \
    < <(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR == n' "$TASKS")
[ -n "$ANALYSIS" ] || { echo "no task row $SLURM_ARRAY_TASK_ID in $TASKS" >&2; exit 1; }
OUT=$ROOT/$ENC/$DATASET/$PART/$ANALYSIS
echo "task $ENC $DATASET $PART $ANALYSIS -> $OUT"
if [ -f "$OUT/DONE" ]; then echo "already done"; exit 0; fi
for f in "$COHORT" $(echo "$CACHES" | tr ' =' '\n\n'); do
    singularity exec "$SIF" test -r "$f" || { echo "not readable in the container: $f" >&2; exit 1; }
done
mkdir -p "$OUT"
N=$SLURM_CPUS_PER_TASK
ENV="export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$N; cd $CODE"
COMMON="--cohort $COHORT --caches $CACHES --out $OUT --fold $PART"
case $ANALYSIS in
    geometry) CMD="python scripts/class_geometry_pilot.py $COMMON --jobs $N $EXTRA && \
python scripts/component_metadata.py --cohort $COHORT --components $OUT/specimen_components.csv \
--modality $OUT/per_class_modality.csv --geometry $OUT/per_class_geometry.csv --out $OUT --fold $PART $META" ;;
    overlap) CMD="python scripts/bhattacharyya_overlap.py $COMMON $EXTRA" ;;
    fda) CMD="python scripts/pca_fda_gbc.py $COMMON $EXTRA" ;;
    lw) CMD="python scripts/lw_fit_diagnostics.py $COMMON $EXTRA" ;;
    coupled) CMD="python scripts/coupled_rscm_vs_lw.py $COMMON $EXTRA" ;;
    pooling) CMD="python scripts/balanced_taxonomic_pooling.py $COMMON $EXTRA" ;;
    *) echo "unknown analysis $ANALYSIS" >&2; exit 1 ;;
esac
singularity exec "$SIF" bash -c "$ENV; $CMD" > "$OUT/run.log" 2> "$OUT/run.err"
date -Is > "$OUT/DONE"
echo "end $(date -Is)"
