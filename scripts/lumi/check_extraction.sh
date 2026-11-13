#!/bin/bash
# CPU checks of one extract_encoder.sh output, run after it (sbatch --dependency=afterok): cache verification against
# the manifest (verify_feature_cache.py -> OUT/verify.json) and a CPU re-extraction of the first 256 manifest rows
# compared with the GPU cache (compare_cache_prefix.py -> OUT/gpu_vs_cpu256.json). Same arguments as extract_encoder.sh.
#
# Usage: sbatch [--time T] check_extraction.sh CODE_DIR ENCODER MANIFEST SOURCE_FLAG SOURCE OUT_DIR
#SBATCH --job-name=fg_extract_check
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

CODE=$1 ENCODER=$2 MANIFEST=$3 SOURCE_FLAG=$4 SOURCE=$5 OUT=$6
source "$CODE/scripts/lumi/container_env.sh"
HF_HOME=/scratch/project_462001534/aqua_v1/scratch_$USER/hf_cache
PKGS=/scratch/project_462001534/feature_geometry/pkgs  # open_clip_torch 3.3.0 (see extract_encoder.sh)
for f in "$CODE/scripts/verify_feature_cache.py" "$SOURCE" "$MANIFEST" "$OUT/features_gpu/$ENCODER"; do
    singularity exec "$SIF" test -r "$f" || { echo "not readable in the container: $f" >&2; exit 1; }
done
ENV="export PYTHONPATH=$CODE:$PKGS HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; cd $CODE"
echo "start $(date -Is) job $SLURM_JOB_ID code $(cat "$CODE/CODE_COMMIT") encoder $ENCODER manifest $MANIFEST out $OUT"
singularity exec "$SIF" bash -c "$ENV; python scripts/verify_feature_cache.py --manifest $MANIFEST \
    --cache-dir $OUT/features_gpu/$ENCODER --encoder $ENCODER --chunk-size 8192 --out $OUT/verify.json"
singularity exec "$SIF" bash -c "$ENV; python scripts/extract_features.py --manifest $MANIFEST $SOURCE_FLAG $SOURCE \
    --encoder $ENCODER --chunk-size 8192 --cache-dir $OUT/features_cpu256 --limit-rows 256 --batch-size 32 \
    --num-workers 4 --accelerator cpu > $OUT/cpucheck.log 2>&1; python scripts/compare_cache_prefix.py \
    --manifest $MANIFEST --full $OUT/features_gpu/$ENCODER --prefix $OUT/features_cpu256/$ENCODER --rows 256 \
    --out $OUT/gpu_vs_cpu256.json"
echo "end $(date -Is)"
