#!/bin/bash
# Frozen image features of one encoder for one manifest on one LUMI node. Eight tasks, one GCD each, encode chunks
# task, task+8, ... (resumable: finished chunks are skipped). Then, on the batch node: cache verification and a CPU
# re-extraction of the first 256 images compared with the GPU cache.
#
# Usage: sbatch [--partition P --time T] extract_encoder.sh CODE_DIR ENCODER MANIFEST SOURCE_FLAG SOURCE OUT_DIR
#   ENCODER: bioclip2 | dinov2_l | dinov3_l | siglip2 (src/encoders/frozen_encoders.py)
#   SOURCE_FLAG SOURCE: --hdf5 FILE.h5 (manifest img + hdf5_row) | --cifar-dir DIR (manifest img = source_id)
#   Features go to OUT_DIR/features_gpu/ENCODER; a rerun with the same OUT_DIR resumes.
#SBATCH --job-name=fg_extract
#SBATCH --account=project_462001534
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail  # no -u: the module scripts read unset variables

export CODE=$1 ENCODER=$2 MANIFEST=$3 SOURCE_FLAG=$4 SOURCE=$5 OUT=$6
source "$CODE/scripts/lumi/container_env.sh"
export HF_HOME=/scratch/project_462001534/aqua_v1/scratch_$USER/hf_cache
mkdir -p "$OUT"
for f in "$CODE/scripts/extract_features.py" "$SOURCE" "$MANIFEST"; do  # fail fast if anything is not visible
    singularity exec "$SIF" test -r "$f" || { echo "not readable in the container: $f" >&2; exit 1; }
done
PKGS=/scratch/project_462001534/feature_geometry/pkgs  # open_clip_torch 3.3.0 (+ ftfy, regex): not in the container; copy of aqua_v1/pkgs
ENV="export PYTHONPATH=$CODE:$PKGS HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1; cd $CODE"
X="python scripts/extract_features.py --manifest $MANIFEST $SOURCE_FLAG $SOURCE --encoder $ENCODER --chunk-size 8192"
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST code $(cat "$CODE/CODE_COMMIT") encoder $ENCODER manifest $MANIFEST out $OUT"
srun --kill-on-bad-exit=1 bash -c "export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID; singularity exec $SIF bash -c \
    '$ENV; $X --cache-dir $OUT/features_gpu --task \$SLURM_PROCID --n-tasks \$SLURM_NTASKS --batch-size 64 \
    --num-workers 6 --accelerator gpu' >> $OUT/extract_task\$SLURM_PROCID.log 2>&1"
echo "extract done $(date -Is)"
singularity exec "$SIF" bash -c "$ENV; python scripts/verify_feature_cache.py --manifest $MANIFEST \
    --cache-dir $OUT/features_gpu/$ENCODER --encoder $ENCODER --chunk-size 8192 --out $OUT/verify.json"
singularity exec "$SIF" bash -c "$ENV; $X --cache-dir $OUT/features_cpu256 --limit-rows 256 --batch-size 32 \
    --num-workers 4 --accelerator cpu > $OUT/cpucheck.log 2>&1; python scripts/compare_cache_prefix.py \
    --manifest $MANIFEST --full $OUT/features_gpu/$ENCODER --prefix $OUT/features_cpu256/$ENCODER --rows 256 \
    --out $OUT/gpu_vs_cpu256.json"
echo "end $(date -Is)"
