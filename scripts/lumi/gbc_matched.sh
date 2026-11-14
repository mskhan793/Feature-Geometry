#!/bin/bash
# Matched-pair GBC comparison (task.md e4d291d) on LUMI CPU: array task i runs scripts/gbc_matched_comparison.py on
# the i-th saved FDA run of the geometry replication (rows with analysis "fda" in
# docs/evidence/geometry_replication/final/summary/task_status.csv: 40 encoder x dataset combinations, 80 runs).
# Reads RUNS/<encoder>/<dataset>/<partition>/fda (saved per-pair tables), writes OUT/<encoder>/<dataset>/<partition>.
# No refitting, no feature or test access.
#
# Usage: sbatch --array=1-80 gbc_matched.sh CODE_DIR RUNS OUT
#SBATCH --job-name=fg_gbcm
#SBATCH --account=project_462001824
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -eo pipefail  # no -u: the module scripts read unset variables

CODE=$1 RUNS=$2 OUT=$3
echo "start $(date -Is) job ${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID code $(cat "$CODE/CODE_COMMIT")"
source "$CODE/scripts/lumi/container_env.sh"
STATUS=$CODE/docs/evidence/geometry_replication/final/summary/task_status.csv
read -r ENC DATASET PART < <(awk -F, '$4 == "fda" {print $1, $2, $3}' "$STATUS" | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$PART" ] || { echo "no fda run $SLURM_ARRAY_TASK_ID in $STATUS" >&2; exit 1; }
SRC=$RUNS/$ENC/$DATASET/$PART/fda
DST=$OUT/$ENC/$DATASET/$PART
echo "run $ENC $DATASET $PART: $SRC -> $DST"
singularity exec "$SIF" test -f "$SRC/DONE" || { echo "no finished FDA run in $SRC" >&2; exit 1; }
mkdir -p "$DST"
singularity exec "$SIF" bash -c "export PYTHONPATH=$CODE PYTHONUNBUFFERED=1; cd $CODE; \
python scripts/gbc_matched_comparison.py --run $SRC \
--support docs/evidence/geometry_replication/inputs/support/${DATASET}_$PART.csv --out $DST \
--encoder $ENC --dataset $DATASET --partition $PART" > "$DST/run.log" 2> "$DST/run.err"
date -Is > "$DST/DONE"
echo "end $(date -Is)"
