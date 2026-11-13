#!/bin/bash
# C3 provenance / QA records (task.md d57a630): one array task per line of RECORDS (records.tsv from
# scripts/benchmark/c2_plan.py, all 32 encoder x dataset combinations) runs scripts/benchmark/combination_record.py
# -> OUT/<encoder>__<dataset>.json (identity vs the standardised manifest, independent units, finite / norms /
# checksum, numerical rank, DINO final-LayerNorm constraint). Read-only on all caches; for the v1 BIOSCAN caches only
# the train and val folders are opened.
# Usage: sbatch --array=0-31 c3_records.sh CODE_DIR RECORDS_TSV OUT_DIR
#SBATCH --job-name=fg_c3_records
#SBATCH --account=project_462001534
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -eo pipefail  # no -u: the module scripts read unset variables

CODE=$1 RECORDS=$2 OUT=$3
source "$CODE/scripts/lumi/container_env.sh"
IFS=$'\t' read -r enc ds manifest standard sources weights < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RECORDS")
mkdir -p "$OUT"
echo "start $(date -Is) job ${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID code $(cat "$CODE/CODE_COMMIT") $enc $ds"
singularity exec "$SIF" bash -c "export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; \
    cd $CODE; python scripts/benchmark/combination_record.py --encoder $enc --dataset $ds --manifest $manifest \
    --standard $standard $sources $weights --out $OUT/${enc}__$ds.json"
echo "end $(date -Is)"
