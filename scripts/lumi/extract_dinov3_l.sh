#!/bin/bash
# Frozen DINOv3-L image features for one manifest of HDF5 images on one LUMI node (FIN-Benthic2 fold-0 cohort,
# task.md 366ee62; full-specimen selections of FIN-Benthic2 and AquaMonitor, task.md 92a605a). Eight tasks, one GCD
# each, encode chunks task, task+8, ... of the manifest (resumable: finished chunks are skipped). Then, on the batch
# node: cache verification and a CPU re-extraction of the first 256 images compared with the GPU cache.
#
# Usage: sbatch [--partition P --time T] extract_dinov3_l.sh CODE_DIR MANIFEST HDF5 OUT_DIR
#   MANIFEST: parquet with img + hdf5_row; features go to OUT_DIR/features_gpu/dinov3_l. A rerun with the same OUT_DIR
#   resumes.
#SBATCH --job-name=fg_extract
#SBATCH --account=project_462001534
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=240G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail  # no -u: the module scripts read unset variables

export CODE=$1 MANIFEST=$2 HDF5=$3 OUT=$4
source "$CODE/scripts/lumi/container_env.sh"
export HF_HOME=/scratch/project_462001534/aqua_v1/scratch_$USER/hf_cache
mkdir -p "$OUT"
for f in "$CODE/scripts/extract_features.py" "$HDF5" "$MANIFEST"; do  # fail fast if anything is not visible
    singularity exec "$SIF" test -r "$f" || { echo "not readable in the container: $f" >&2; exit 1; }
done
ENV="export PYTHONPATH=$CODE HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1; cd $CODE"
X="python scripts/extract_features.py --manifest $MANIFEST --hdf5 $HDF5 --encoder dinov3_l --chunk-size 8192"
echo "start $(date -Is) job $SLURM_JOB_ID node $SLURM_NODELIST code $(cat "$CODE/CODE_COMMIT") manifest $MANIFEST out $OUT"
srun --kill-on-bad-exit=1 bash -c "export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID; singularity exec $SIF bash -c \
    '$ENV; $X --cache-dir $OUT/features_gpu --task \$SLURM_PROCID --n-tasks \$SLURM_NTASKS --batch-size 64 \
    --num-workers 6 --accelerator gpu' >> $OUT/extract_task\$SLURM_PROCID.log 2>&1"
echo "extract done $(date -Is)"
singularity exec "$SIF" bash -c "$ENV; python scripts/verify_feature_cache.py --manifest $MANIFEST \
    --cache-dir $OUT/features_gpu/dinov3_l --encoder dinov3_l --chunk-size 8192 --out $OUT/verify.json"
singularity exec "$SIF" bash -c "$ENV; $X --cache-dir $OUT/features_cpu256 --limit-rows 256 --batch-size 32 \
    --num-workers 4 --accelerator cpu > $OUT/cpucheck.log 2>&1; python scripts/compare_cache_prefix.py \
    --manifest $MANIFEST --full $OUT/features_gpu/dinov3_l --prefix $OUT/features_cpu256/dinov3_l --rows 256 \
    --out $OUT/gpu_vs_cpu256.json"
echo "end $(date -Is)"
